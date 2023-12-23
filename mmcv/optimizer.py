# Copyright (c) OpenMMLab. All rights reserved.
import copy
import logging
from collections import defaultdict
from itertools import chain
from typing import Optional, Union

import torch
import torch.nn as nn
import numpy as np
from torch import Tensor
from torch.nn.utils import clip_grad
from torch.autograd import Variable

from mmcv.utils import (IS_NPU_AVAILABLE, TORCH_VERSION, _BatchNorm,
                        digit_version)
from ..dist_utils import allreduce_grads
from ..fp16_utils import LossScaler, wrap_fp16_model
from .hook import HOOKS, Hook

try:
    # If PyTorch version >= 1.6.0, torch.cuda.amp.GradScaler would be imported
    # and used; otherwise, auto fp16 will adopt mmcv's implementation.
    if IS_NPU_AVAILABLE:
        from torch.npu.amp import GradScaler
    else:
        from torch.cuda.amp import GradScaler
except ImportError:
    pass


class MinNormSolver:
    MAX_ITER = 250
    STOP_CRIT = 1e-5

    def _min_norm_element_from2(v1v1, v1v2, v2v2):
        """
        Analytical solution for min_{c} |cx_1 + (1-c)x_2|_2^2
        d is the distance (objective) optimzed
        v1v1 = <x1,x1>
        v1v2 = <x1,x2>
        v2v2 = <x2,x2>
        """
        if v1v2 >= v1v1:
            # Case: Fig 1, third column
            gamma = 0.999
            cost = v1v1
            return gamma, cost
        if v1v2 >= v2v2:
            # Case: Fig 1, first column
            gamma = 0.001
            cost = v2v2
            return gamma, cost
        # Case: Fig 1, second column
        gamma = -1.0 * ((v1v2 - v2v2) / (v1v1 + v2v2 - 2 * v1v2))
        cost = v2v2 + gamma * (v1v2 - v2v2)
        return gamma, cost

    def _min_norm_2d(vecs, dps):
        """
        Find the minimum norm solution as combination of two points
        This is correct only in 2D
        ie. min_c |\sum c_i x_i|_2^2 st. \sum c_i = 1 , 1 >= c_1 >= 0 for all i, c_i + c_j = 1.0 for some i, j
        """
        dmin = 1e8
        for i in range(len(vecs)):
            for j in range(i + 1, len(vecs)):
                if (i, j) not in dps:
                    dps[(i, j)] = 0.0
                    for k in range(len(vecs[i])):
                        dps[(i, j)] += torch.mul(vecs[i][k], vecs[j][k]).sum().data.cpu()
                    dps[(j, i)] = dps[(i, j)]
                if (i, i) not in dps:
                    dps[(i, i)] = 0.0
                    for k in range(len(vecs[i])):
                        dps[(i, i)] += torch.mul(vecs[i][k], vecs[i][k]).sum().data.cpu()
                if (j, j) not in dps:
                    dps[(j, j)] = 0.0
                    for k in range(len(vecs[i])):
                        dps[(j, j)] += torch.mul(vecs[j][k], vecs[j][k]).sum().data.cpu()
                c, d = MinNormSolver._min_norm_element_from2(dps[(i, i)], dps[(i, j)], dps[(j, j)])
                # if d < dmin:
                dmin = d
                sol = [(i, j), c, d]
        return sol, dps

    def _projection2simplex(y):
        """
        Given y, it solves argmin_z |y-z|_2 st \sum z = 1 , 1 >= z_i >= 0 for all i
        """
        m = len(y)
        sorted_y = np.flip(np.sort(y), axis=0)
        tmpsum = 0.0
        tmax_f = (np.sum(y) - 1.0) / m
        for i in range(m - 1):
            tmpsum += sorted_y[i]
            tmax = (tmpsum - 1) / (i + 1.0)
            if tmax > sorted_y[i + 1]:
                tmax_f = tmax
                break
        return np.maximum(y - tmax_f, np.zeros(y.shape))

    def _next_point(cur_val, grad, n):
        proj_grad = grad - (np.sum(grad) / n)
        tm1 = -1.0 * cur_val[proj_grad < 0] / proj_grad[proj_grad < 0]
        tm2 = (1.0 - cur_val[proj_grad > 0]) / (proj_grad[proj_grad > 0])

        skippers = np.sum(tm1 < 1e-7) + np.sum(tm2 < 1e-7)
        t = 1
        if len(tm1[tm1 > 1e-7]) > 0:
            t = np.min(tm1[tm1 > 1e-7])
        if len(tm2[tm2 > 1e-7]) > 0:
            t = min(t, np.min(tm2[tm2 > 1e-7]))

        next_point = proj_grad * t + cur_val
        next_point = MinNormSolver._projection2simplex(next_point)
        return next_point

    def find_min_norm_element(vecs):
        """
        Given a list of vectors (vecs), this method finds the minimum norm element in the convex hull
        as min |u|_2 st. u = \sum c_i vecs[i] and \sum c_i = 1.
        It is quite geometric, and the main idea is the fact that if d_{ij} = min |u|_2 st u = c x_i + (1-c) x_j; the solution lies in (0, d_{i,j})
        Hence, we find the best 2-task solution, and then run the projected gradient descent until convergence
        """
        # Solution lying at the combination of two points
        dps = {}
        init_sol, dps = MinNormSolver._min_norm_2d(vecs, dps)

        n = len(vecs)
        sol_vec = np.zeros(n)
        sol_vec[init_sol[0][0]] = abs(init_sol[1])
        sol_vec[init_sol[0][1]] = 1 - abs(init_sol[1])

        if n < 3:
            # This is optimal for n=2, so return the solution
            return sol_vec, init_sol[2]

        iter_count = 0

        grad_mat = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                grad_mat[i, j] = dps[(i, j)]

        while iter_count < MinNormSolver.MAX_ITER:
            grad_dir = -1.0 * np.dot(grad_mat, sol_vec)
            new_point = MinNormSolver._next_point(sol_vec, grad_dir, n)
            # Re-compute the inner products for line search
            v1v1 = 0.0
            v1v2 = 0.0
            v2v2 = 0.0
            for i in range(n):
                for j in range(n):
                    v1v1 += sol_vec[i] * sol_vec[j] * dps[(i, j)]
                    v1v2 += sol_vec[i] * new_point[j] * dps[(i, j)]
                    v2v2 += new_point[i] * new_point[j] * dps[(i, j)]
            nc, nd = MinNormSolver._min_norm_element_from2(v1v1, v1v2, v2v2)
            new_sol_vec = nc * sol_vec + (1 - nc) * new_point
            change = new_sol_vec - sol_vec
            if np.sum(np.abs(change)) < MinNormSolver.STOP_CRIT:
                return sol_vec, nd
            sol_vec = new_sol_vec

    def find_min_norm_element_FW(vecs):
        """
        Given a list of vectors (vecs), this method finds the minimum norm element in the convex hull
        as min |u|_2 st. u = \sum c_i vecs[i] and \sum c_i = 1.
        It is quite geometric, and the main idea is the fact that if d_{ij} = min |u|_2 st u = c x_i + (1-c) x_j; the solution lies in (0, d_{i,j})
        Hence, we find the best 2-task solution, and then run the Frank Wolfe until convergence
        """
        # Solution lying at the combination of two points
        dps = {}
        init_sol, dps = MinNormSolver._min_norm_2d(vecs, dps)

        n = len(vecs)
        sol_vec = np.zeros(n)
        sol_vec[init_sol[0][0]] = init_sol[1]
        sol_vec[init_sol[0][1]] = 1 - init_sol[1]

        if n < 3:
            # This is optimal for n=2, so return the solution
            return sol_vec, init_sol[2]

        iter_count = 0

        grad_mat = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                grad_mat[i, j] = dps[(i, j)]

        while iter_count < MinNormSolver.MAX_ITER:
            t_iter = np.argmin(np.dot(grad_mat, sol_vec))

            v1v1 = np.dot(sol_vec, np.dot(grad_mat, sol_vec))
            v1v2 = np.dot(sol_vec, grad_mat[:, t_iter])
            v2v2 = grad_mat[t_iter, t_iter]

            nc, nd = MinNormSolver._min_norm_element_from2(v1v1, v1v2, v2v2)
            new_sol_vec = nc * sol_vec
            new_sol_vec[t_iter] += 1 - nc

            change = new_sol_vec - sol_vec
            if np.sum(np.abs(change)) < MinNormSolver.STOP_CRIT:
                return sol_vec, nd
            sol_vec = new_sol_vec

def gradient_normalizers(grads, losses, normalization_type):
    gn = {}
    if normalization_type == 'l2':
        for t in grads:
            gn[t] = np.sqrt(np.sum([gr.pow(2).sum().data.cpu() for gr in grads[t]]))
    elif normalization_type == 'loss':
        for t in grads:
            gn[t] = losses[t]
    elif normalization_type == 'loss+':
        for t in grads:
            gn[t] = losses[t] * np.sqrt(np.sum([gr.pow(2).sum().data.cpu() for gr in grads[t]]))
    elif normalization_type == 'none':
        for t in grads:
            gn[t] = 1.0
    else:
        print('ERROR: Invalid Normalization Type')
    return gn

@HOOKS.register_module()
class OptimizerHook(Hook):
    """A hook contains custom operations for the optimizer.

    Args:
        grad_clip (dict, optional): A config dict to control the clip_grad.
            Default: None.
        detect_anomalous_params (bool): This option is only used for
            debugging which will slow down the training speed.
            Detect anomalous parameters that are not included in
            the computational graph with `loss` as the root.
            There are two cases

                - Parameters were not used during
                  forward pass.
                - Parameters were not used to produce
                  loss.
            Default: False.
    """

    def __init__(self,
                 grad_clip: Optional[dict] = None,
                 detect_anomalous_params: bool = False):
        self.grad_clip = grad_clip
        self.detect_anomalous_params = detect_anomalous_params

    def clip_grads(self, params):
        params = list(
            filter(lambda p: p.requires_grad and p.grad is not None, params))
        if len(params) > 0:
            return clip_grad.clip_grad_norm_(params, **self.grad_clip)

    def after_train_iter(self, runner):
        grads = {}
        loss_data = {}
        scale = {}
        # loss1 + grad1
        runner.optimizer.zero_grad()
        if self.detect_anomalous_params:
            self.detect_anomalous_parameters(runner.outputs['loss'], runner)
        runner.outputs['loss'][0].backward(retain_graph=True)
        grad1 = []
        for name, param in runner.model.named_parameters():
            if name == 'module.cls_head.fc_cls.weight' or name == 'module.cls_head.fc_cls.bias':
                grad1.append(Variable(param.grad.data.clone(), requires_grad=False))
        grads[0] = grad1
        loss_data[0] = runner.outputs['loss'][0]

        # loss2 + grad2
        runner.optimizer.zero_grad()
        if self.detect_anomalous_params:
            self.detect_anomalous_parameters(runner.outputs['loss'], runner)
        runner.outputs['loss'][1].backward(retain_graph=True)
        grad2 = []
        for name, param in runner.model.named_parameters():
            if name == 'module.cls_head2.fc_cls.weight' or name == 'module.cls_head2.fc_cls.bias':
                grad2.append(Variable(param.grad.data.clone(), requires_grad=False))
        grads[1] = grad2
        loss_data[1] = runner.outputs['loss'][1]

        # Normalize all gradients, this is optional and not included in the paper.
        gn = gradient_normalizers(grads, loss_data, 'loss+')
        for gr_i in range(len(grads[0])):
            grads[0][gr_i] = grads[0][gr_i] / gn[0]
        for gr_i in range(len(grads[1])):
            grads[1][gr_i] = grads[1][gr_i] / gn[1]

        # Frank-Wolfe iteration to compute scales.
        sol, min_norm = MinNormSolver.find_min_norm_element([grads[0], grads[1]])
        scale[0] = float(sol[0])
        scale[1] = float(sol[1])

        runner.optimizer.zero_grad()
        pareto_loss = 10*(scale[0]*loss_data[0] + scale[1]*loss_data[1])
        pareto_loss.backward()
        if self.grad_clip is not None:
            grad_norm = self.clip_grads(runner.model.parameters())
            if grad_norm is not None:
                # Add grad norm to the logger
                runner.log_buffer.update({'grad_norm': float(grad_norm)},
                                         runner.outputs['num_samples'])
        runner.optimizer.step()

    def detect_anomalous_parameters(self, loss: Tensor, runner) -> None:
        logger = runner.logger
        parameters_in_graph = set()
        visited = set()

        def traverse(grad_fn):
            if grad_fn is None:
                return
            if grad_fn not in visited:
                visited.add(grad_fn)
                if hasattr(grad_fn, 'variable'):
                    parameters_in_graph.add(grad_fn.variable)
                parents = grad_fn.next_functions
                if parents is not None:
                    for parent in parents:
                        grad_fn = parent[0]
                        traverse(grad_fn)

        traverse(loss.grad_fn)
        for n, p in runner.model.named_parameters():
            if p not in parameters_in_graph and p.requires_grad:
                logger.log(
                    level=logging.ERROR,
                    msg=f'{n} with shape {p.size()} is not '
                    f'in the computational graph \n')


@HOOKS.register_module()
class GradientCumulativeOptimizerHook(OptimizerHook):
    """Optimizer Hook implements multi-iters gradient cumulating.

    Args:
        cumulative_iters (int, optional): Num of gradient cumulative iters.
            The optimizer will step every `cumulative_iters` iters.
            Defaults to 1.

    Examples:
        >>> # Use cumulative_iters to simulate a large batch size
        >>> # It is helpful when the hardware cannot handle a large batch size.
        >>> loader = DataLoader(data, batch_size=64)
        >>> optim_hook = GradientCumulativeOptimizerHook(cumulative_iters=4)
        >>> # almost equals to
        >>> loader = DataLoader(data, batch_size=256)
        >>> optim_hook = OptimizerHook()
    """

    def __init__(self, cumulative_iters: int = 1, **kwargs):
        super().__init__(**kwargs)

        assert isinstance(cumulative_iters, int) and cumulative_iters > 0, \
            f'cumulative_iters only accepts positive int, but got ' \
            f'{type(cumulative_iters)} instead.'

        self.cumulative_iters = cumulative_iters
        self.divisible_iters = 0
        self.remainder_iters = 0
        self.initialized = False

    def has_batch_norm(self, module: nn.Module) -> bool:
        if isinstance(module, _BatchNorm):
            return True
        for m in module.children():
            if self.has_batch_norm(m):
                return True
        return False

    def _init(self, runner):
        if runner.iter % self.cumulative_iters != 0:
            runner.logger.warning(
                'Resume iter number is not divisible by cumulative_iters in '
                'GradientCumulativeOptimizerHook, which means the gradient of '
                'some iters is lost and the result may be influenced slightly.'
            )

        if self.has_batch_norm(runner.model) and self.cumulative_iters > 1:
            runner.logger.warning(
                'GradientCumulativeOptimizerHook may slightly decrease '
                'performance if the model has BatchNorm layers.')

        self.divisible_iters = (
            runner.max_iters // self.cumulative_iters * self.cumulative_iters)
        self.remainder_iters = runner.max_iters - self.divisible_iters

        self.initialized = True

    def _get_loss_factor(self, runner):
        """Get loss division factor for the current iteration."""
        if runner.iter < runner.max_iters - self.remainder_iters:
            loss_factor = self.cumulative_iters
        else:
            loss_factor = self.remainder_iters
            runner.logger.warning(
                f'Loss will be divided by {loss_factor} in the last '
                f'{self.remainder_iters} iterations because they are not '
                f'enough for {self.cumulative_iters} cumulative_iters.')
            assert loss_factor > 0

        return loss_factor

    def after_train_iter(self, runner):
        if not self.initialized:
            self._init(runner)

        loss = runner.outputs['loss'] / self._get_loss_factor(runner)
        loss.backward()

        if (self.every_n_iters(runner, self.cumulative_iters)
                or self.is_last_iter(runner)):

            if self.grad_clip is not None:
                grad_norm = self.clip_grads(runner.model.parameters())
                if grad_norm is not None:
                    # Add grad norm to the logger
                    runner.log_buffer.update({'grad_norm': float(grad_norm)},
                                             runner.outputs['num_samples'])
            runner.optimizer.step()
            runner.optimizer.zero_grad()


if (TORCH_VERSION != 'parrots'
        and digit_version(TORCH_VERSION) >= digit_version('1.6.0')):

    @HOOKS.register_module()
    class Fp16OptimizerHook(OptimizerHook):
        """FP16 optimizer hook (using PyTorch's implementation).

        If you are using PyTorch >= 1.6, torch.cuda.amp is used as the backend,
        to take care of the optimization procedure.

        Args:
            loss_scale (float | str | dict): Scale factor configuration.
                If loss_scale is a float, static loss scaling will be used with
                the specified scale. If loss_scale is a string, it must be
                'dynamic', then dynamic loss scaling will be used.
                It can also be a dict containing arguments of GradScalar.
                Defaults to 512. For Pytorch >= 1.6, mmcv uses official
                implementation of GradScaler. If you use a dict version of
                loss_scale to create GradScaler, please refer to:
                https://pytorch.org/docs/stable/amp.html#torch.cuda.amp.GradScaler
                for the parameters.

        Examples:
            >>> loss_scale = dict(
            ...     init_scale=65536.0,
            ...     growth_factor=2.0,
            ...     backoff_factor=0.5,
            ...     growth_interval=2000
            ... )
            >>> optimizer_hook = Fp16OptimizerHook(loss_scale=loss_scale)
        """

        def __init__(self,
                     grad_clip: Optional[dict] = None,
                     coalesce: bool = True,
                     bucket_size_mb: int = -1,
                     loss_scale: Union[float, str, dict] = 512.,
                     distributed: bool = True):
            self.grad_clip = grad_clip
            self.coalesce = coalesce
            self.bucket_size_mb = bucket_size_mb
            self.distributed = distributed
            self._scale_update_param = None
            if loss_scale == 'dynamic':
                self.loss_scaler = GradScaler()
            elif isinstance(loss_scale, float):
                self._scale_update_param = loss_scale
                self.loss_scaler = GradScaler(init_scale=loss_scale)
            elif isinstance(loss_scale, dict):
                self.loss_scaler = GradScaler(**loss_scale)
            else:
                raise ValueError('loss_scale must be of type float, dict, or '
                                 f'"dynamic", got {loss_scale}')

        def before_run(self, runner) -> None:
            """Preparing steps before Mixed Precision Training."""
            # wrap model mode to fp16
            wrap_fp16_model(runner.model)
            # resume from state dict
            if 'fp16' in runner.meta and 'loss_scaler' in runner.meta['fp16']:
                scaler_state_dict = runner.meta['fp16']['loss_scaler']
                self.loss_scaler.load_state_dict(scaler_state_dict)

        def copy_grads_to_fp32(self, fp16_net: nn.Module,
                               fp32_weights: Tensor) -> None:
            """Copy gradients from fp16 model to fp32 weight copy."""
            for fp32_param, fp16_param in zip(fp32_weights,
                                              fp16_net.parameters()):
                if fp16_param.grad is not None:
                    if fp32_param.grad is None:
                        fp32_param.grad = fp32_param.data.new(
                            fp32_param.size())
                    fp32_param.grad.copy_(fp16_param.grad)

        def copy_params_to_fp16(self, fp16_net: nn.Module,
                                fp32_weights: Tensor) -> None:
            """Copy updated params from fp32 weight copy to fp16 model."""
            for fp16_param, fp32_param in zip(fp16_net.parameters(),
                                              fp32_weights):
                fp16_param.data.copy_(fp32_param.data)

        def after_train_iter(self, runner) -> None:
            """Backward optimization steps for Mixed Precision Training. For
            dynamic loss scaling, please refer to
            https://pytorch.org/docs/stable/amp.html#torch.cuda.amp.GradScaler.

            1. Scale the loss by a scale factor.
            2. Backward the loss to obtain the gradients.
            3. Unscale the optimizer’s gradient tensors.
            4. Call optimizer.step() and update scale factor.
            5. Save loss_scaler state_dict for resume purpose.
            """
            # clear grads of last iteration
            runner.model.zero_grad()
            runner.optimizer.zero_grad()

            self.loss_scaler.scale(runner.outputs['loss']).backward()
            self.loss_scaler.unscale_(runner.optimizer)
            # grad clip
            if self.grad_clip is not None:
                grad_norm = self.clip_grads(runner.model.parameters())
                if grad_norm is not None:
                    # Add grad norm to the logger
                    runner.log_buffer.update({'grad_norm': float(grad_norm)},
                                             runner.outputs['num_samples'])
            # backward and update scaler
            self.loss_scaler.step(runner.optimizer)
            self.loss_scaler.update(self._scale_update_param)

            # save state_dict of loss_scaler
            runner.meta.setdefault(
                'fp16', {})['loss_scaler'] = self.loss_scaler.state_dict()

    @HOOKS.register_module()
    class GradientCumulativeFp16OptimizerHook(GradientCumulativeOptimizerHook,
                                              Fp16OptimizerHook):
        """Fp16 optimizer Hook (using PyTorch's implementation) implements
        multi-iters gradient cumulating.

        If you are using PyTorch >= 1.6, torch.cuda.amp is used as the backend,
        to take care of the optimization procedure.
        """

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)

        def after_train_iter(self, runner) -> None:
            if not self.initialized:
                self._init(runner)

            loss = runner.outputs['loss'] / self._get_loss_factor(runner)
            self.loss_scaler.scale(loss).backward()

            if (self.every_n_iters(runner, self.cumulative_iters)
                    or self.is_last_iter(runner)):

                # copy fp16 grads in the model to fp32 params in the optimizer
                self.loss_scaler.unscale_(runner.optimizer)

                if self.grad_clip is not None:
                    grad_norm = self.clip_grads(runner.model.parameters())
                    if grad_norm is not None:
                        # Add grad norm to the logger
                        runner.log_buffer.update(
                            {'grad_norm': float(grad_norm)},
                            runner.outputs['num_samples'])

                # backward and update scaler
                self.loss_scaler.step(runner.optimizer)
                self.loss_scaler.update(self._scale_update_param)

                # save state_dict of loss_scaler
                runner.meta.setdefault(
                    'fp16', {})['loss_scaler'] = self.loss_scaler.state_dict()

                # clear grads
                runner.model.zero_grad()
                runner.optimizer.zero_grad()

else:

    @HOOKS.register_module()
    class Fp16OptimizerHook(OptimizerHook):  # type: ignore
        """FP16 optimizer hook (mmcv's implementation).

        The steps of fp16 optimizer is as follows.
        1. Scale the loss value.
        2. BP in the fp16 model.
        2. Copy gradients from fp16 model to fp32 weights.
        3. Update fp32 weights.
        4. Copy updated parameters from fp32 weights to fp16 model.

        Refer to https://arxiv.org/abs/1710.03740 for more details.

        Args:
            loss_scale (float | str | dict): Scale factor configuration.
                If loss_scale is a float, static loss scaling will be used with
                the specified scale. If loss_scale is a string, it must be
                'dynamic', then dynamic loss scaling will be used.
                It can also be a dict containing arguments of LossScaler.
                Defaults to 512.
        """

        def __init__(self,
                     grad_clip: Optional[dict] = None,
                     coalesce: bool = True,
                     bucket_size_mb: int = -1,
                     loss_scale: Union[float, str, dict] = 512.,
                     distributed: bool = True):
            self.grad_clip = grad_clip
            self.coalesce = coalesce
            self.bucket_size_mb = bucket_size_mb
            self.distributed = distributed
            if loss_scale == 'dynamic':
                self.loss_scaler = LossScaler(mode='dynamic')
            elif isinstance(loss_scale, float):
                self.loss_scaler = LossScaler(
                    init_scale=loss_scale, mode='static')
            elif isinstance(loss_scale, dict):
                self.loss_scaler = LossScaler(**loss_scale)
            else:
                raise ValueError('loss_scale must be of type float, dict, or '
                                 f'"dynamic", got {loss_scale}')

        def before_run(self, runner) -> None:
            """Preparing steps before Mixed Precision Training.

            1. Make a master copy of fp32 weights for optimization.
            2. Convert the main model from fp32 to fp16.
            """
            # keep a copy of fp32 weights
            old_groups = runner.optimizer.param_groups
            runner.optimizer.param_groups = copy.deepcopy(
                runner.optimizer.param_groups)
            state: defaultdict = defaultdict(dict)
            p_map = {
                old_p: p
                for old_p, p in zip(
                    chain(*(g['params'] for g in old_groups)),
                    chain(*(g['params']
                            for g in runner.optimizer.param_groups)))
            }
            for k, v in runner.optimizer.state.items():
                state[p_map[k]] = v
            runner.optimizer.state = state
            # convert model to fp16
            wrap_fp16_model(runner.model)
            # resume from state dict
            if 'fp16' in runner.meta and 'loss_scaler' in runner.meta['fp16']:
                scaler_state_dict = runner.meta['fp16']['loss_scaler']
                self.loss_scaler.load_state_dict(scaler_state_dict)

        def copy_grads_to_fp32(self, fp16_net: nn.Module,
                               fp32_weights: Tensor) -> None:
            """Copy gradients from fp16 model to fp32 weight copy."""
            for fp32_param, fp16_param in zip(fp32_weights,
                                              fp16_net.parameters()):
                if fp16_param.grad is not None:
                    if fp32_param.grad is None:
                        fp32_param.grad = fp32_param.data.new(
                            fp32_param.size())
                    fp32_param.grad.copy_(fp16_param.grad)

        def copy_params_to_fp16(self, fp16_net: nn.Module,
                                fp32_weights: Tensor) -> None:
            """Copy updated params from fp32 weight copy to fp16 model."""
            for fp16_param, fp32_param in zip(fp16_net.parameters(),
                                              fp32_weights):
                fp16_param.data.copy_(fp32_param.data)

        def after_train_iter(self, runner) -> None:
            """Backward optimization steps for Mixed Precision Training. For
            dynamic loss scaling, please refer `loss_scalar.py`

            1. Scale the loss by a scale factor.
            2. Backward the loss to obtain the gradients (fp16).
            3. Copy gradients from the model to the fp32 weight copy.
            4. Scale the gradients back and update the fp32 weight copy.
            5. Copy back the params from fp32 weight copy to the fp16 model.
            6. Save loss_scaler state_dict for resume purpose.
            """
            # clear grads of last iteration
            runner.model.zero_grad()
            runner.optimizer.zero_grad()
            # scale the loss value
            scaled_loss = runner.outputs['loss'] * self.loss_scaler.loss_scale
            scaled_loss.backward()
            # copy fp16 grads in the model to fp32 params in the optimizer

            fp32_weights = []
            for param_group in runner.optimizer.param_groups:
                fp32_weights += param_group['params']
            self.copy_grads_to_fp32(runner.model, fp32_weights)
            # allreduce grads
            if self.distributed:
                allreduce_grads(fp32_weights, self.coalesce,
                                self.bucket_size_mb)

            has_overflow = self.loss_scaler.has_overflow(fp32_weights)
            # if has overflow, skip this iteration
            if not has_overflow:
                # scale the gradients back
                for param in fp32_weights:
                    if param.grad is not None:
                        param.grad.div_(self.loss_scaler.loss_scale)
                if self.grad_clip is not None:
                    grad_norm = self.clip_grads(fp32_weights)
                    if grad_norm is not None:
                        # Add grad norm to the logger
                        runner.log_buffer.update(
                            {'grad_norm': float(grad_norm)},
                            runner.outputs['num_samples'])
                # update fp32 params
                runner.optimizer.step()
                # copy fp32 params to the fp16 model
                self.copy_params_to_fp16(runner.model, fp32_weights)
            self.loss_scaler.update_scale(has_overflow)
            if has_overflow:
                runner.logger.warning('Check overflow, downscale loss scale '
                                      f'to {self.loss_scaler.cur_scale}')

            # save state_dict of loss_scaler
            runner.meta.setdefault(
                'fp16', {})['loss_scaler'] = self.loss_scaler.state_dict()

    @HOOKS.register_module()
    class GradientCumulativeFp16OptimizerHook(  # type: ignore
            GradientCumulativeOptimizerHook, Fp16OptimizerHook):
        """Fp16 optimizer Hook (using mmcv implementation) implements multi-
        iters gradient cumulating."""

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)

        def after_train_iter(self, runner) -> None:
            if not self.initialized:
                self._init(runner)

            loss = runner.outputs['loss'] / self._get_loss_factor(runner)
            scaled_loss = loss * self.loss_scaler.loss_scale
            scaled_loss.backward()

            if (self.every_n_iters(runner, self.cumulative_iters)
                    or self.is_last_iter(runner)):

                # copy fp16 grads in the model to fp32 params in the optimizer
                fp32_weights = []
                for param_group in runner.optimizer.param_groups:
                    fp32_weights += param_group['params']
                self.copy_grads_to_fp32(runner.model, fp32_weights)
                # allreduce grads
                if self.distributed:
                    allreduce_grads(fp32_weights, self.coalesce,
                                    self.bucket_size_mb)

                has_overflow = self.loss_scaler.has_overflow(fp32_weights)
                # if has overflow, skip this iteration
                if not has_overflow:
                    # scale the gradients back
                    for param in fp32_weights:
                        if param.grad is not None:
                            param.grad.div_(self.loss_scaler.loss_scale)
                    if self.grad_clip is not None:
                        grad_norm = self.clip_grads(fp32_weights)
                        if grad_norm is not None:
                            # Add grad norm to the logger
                            runner.log_buffer.update(
                                {'grad_norm': float(grad_norm)},
                                runner.outputs['num_samples'])
                    # update fp32 params
                    runner.optimizer.step()
                    # copy fp32 params to the fp16 model
                    self.copy_params_to_fp16(runner.model, fp32_weights)
                else:
                    runner.logger.warning(
                        'Check overflow, downscale loss scale '
                        f'to {self.loss_scaler.cur_scale}')

                self.loss_scaler.update_scale(has_overflow)

                # save state_dict of loss_scaler
                runner.meta.setdefault(
                    'fp16', {})['loss_scaler'] = self.loss_scaler.state_dict()

                # clear grads
                runner.model.zero_grad()
                runner.optimizer.zero_grad()
