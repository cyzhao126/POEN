import argparse
import os
import os.path as osp
import warnings
import sys
import numpy as np
import mmcv
import csv
import torch
from mmcv import Config, DictAction
from mmcv.cnn import fuse_conv_bn
from mmcv.fileio.io import file_handlers
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import get_dist_info, init_dist, load_checkpoint
from mmcv.runner.fp16_utils import wrap_fp16_model

from mmaction.datasets import build_dataloader, build_dataset
from mmaction.models import build_model
from mmaction.utils import register_module_hooks

# TODO import test functions from mmcv and delete them from mmaction2
try:
    from mmcv.engine import multi_gpu_test, single_gpu_test
except (ImportError, ModuleNotFoundError):
    warnings.warn(
        'DeprecationWarning: single_gpu_test, multi_gpu_test, '
        'collect_results_cpu, collect_results_gpu from mmaction2 will be '
        'deprecated. Please install mmcv through master branch.')
    from mmaction.apis import multi_gpu_test, single_gpu_test


def parse_args():
    parser = argparse.ArgumentParser(
        description='MMAction2 test (and eval) a model')
    parser.add_argument('--config',
                        default='C:/zcy/Video-Swin-Transformer-master/Video-Swin-Transformer-master/configs/recognition/swin/track1-top2.py',
                        help='test config file path')
    parser.add_argument('--checkpoint',
                        default='C:/zcy/video_swin_transformer_ssh/data/TRACK1/baseline_batch_8_kinetics400/epoch_5.pth',
                        help='checkpoint file')
    parser.add_argument(
        '--out',
        default=None,
        help='output result file in pkl/yaml/json format')
    parser.add_argument(
        '--fuse-conv-bn',
        action='store_true',
        help='Whether to fuse conv and bn, this will slightly increase'
        'the inference speed')
    parser.add_argument(
        '--eval',
        default="mean_average_precision",
        type=str,
        nargs='+',
        help='evaluation metrics, which depends on the dataset, e.g.,'
        ' "top_k_accuracy", "mean_class_accuracy" for video dataset')
    parser.add_argument(
        '--gpu-collect',
        action='store_true',
        help='whether to use gpu to collect results')
    parser.add_argument(
        '--tmpdir',
        help='tmp directory used for collecting results from multiple '
        'workers, available when gpu-collect is not specified')
    parser.add_argument(
        '--options',
        nargs='+',
        action=DictAction,
        default={},
        help='custom options for evaluation, the key-value pair in xxx=yyy '
        'format will be kwargs for dataset.evaluate() function (deprecate), '
        'change to --eval-options instead.')
    parser.add_argument(
        '--eval-options',
        nargs='+',
        action=DictAction,
        default={},
        help='custom options for evaluation, the key-value pair in xxx=yyy '
        'format will be kwargs for dataset.evaluate() function')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        default={},
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. For example, '
        "'--cfg-options model.backbone.depth=18 model.backbone.with_cp=True'")
    parser.add_argument(
        '--average-clips',
        choices=['score', 'prob', None],
        default=None,
        help='average type when averaging test clips')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument(
        '--onnx',
        action='store_true',
        help='Whether to test with onnx model or not')
    parser.add_argument(
        '--tensorrt',
        action='store_true',
        help='Whether to test with TensorRT engine or not')
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    if args.options and args.eval_options:
        raise ValueError(
            '--options and --eval-options cannot be both '
            'specified, --options is deprecated in favor of --eval-options')
    if args.options:
        warnings.warn('--options is deprecated in favor of --eval-options')
        args.eval_options = args.options
    return args

def average_precision(scores, target, max_k=None):
    assert scores.shape == target.shape, "The input and targets do not have the same shape"
    assert scores.ndim == 1, "The input has dimension {}, but expected it to be 1D".format(scores.shape)

    # sort examples
    indices = np.argsort(scores, axis=0)[::-1]

    total_cases = np.sum(target)

    if max_k == None:
        max_k = len(indices)

    # Computes prec@i
    pos_count = 0.
    total_count = 0.
    precision_at_i = 0.

    for i in range(max_k):
        label = target[indices[i]]
        total_count += 1
        if label == 1:
            pos_count += 1
            precision_at_i += pos_count / total_count
        if pos_count == total_cases:
            break

    if pos_count > 0:
        precision_at_i /= pos_count
    else:
        precision_at_i = 0
    return precision_at_i


def micro_f1(Ng, Np, Nc):
    mF1 = (2 * np.sum(Nc)) / (np.sum(Np) + np.sum(Ng))

    return mF1


def macro_f1(Ng, Np, Nc):
    n_class = len(Ng)
    precision_k = Nc / Np
    recall_k = Nc / Ng
    F1_k = (2 * precision_k * recall_k) / (precision_k + recall_k)

    F1_k[np.isnan(F1_k)] = 0

    MF1 = np.sum(F1_k) / n_class

    return precision_k, recall_k, F1_k, MF1


def overall_metrics(Ng, Np, Nc):
    OP = np.sum(Nc) / np.sum(Np)
    OR = np.sum(Nc) / np.sum(Ng)
    OF1 = (2 * OP * OR) / (OP + OR)

    return OP, OR, OF1


def per_class_metrics(Ng, Np, Nc):
    n_class = len(Ng)
    CP = np.sum(Nc / Np) / n_class
    CR = np.sum(Nc / Ng) / n_class
    CF1 = (2 * CP * CR) / (CP + CR)

    return CP, CR, CF1


def mean_average_precision(ap):
    return np.mean(ap)


def exact_match_accuracy(scores, targets, threshold=0.5):
    n_examples, n_class = scores.shape

    binary_mat = np.equal(targets, (scores >= threshold))
    row_sums = binary_mat.sum(axis=1)

    perfect_match = np.zeros(row_sums.shape)
    perfect_match[row_sums == n_class] = 1

    EMAcc = np.sum(perfect_match) / n_examples

    return EMAcc


def class_weighted_f2(Ng, Np, Nc, weights, threshold=0.5):
    n_class = len(Ng)
    precision_k = Nc / Np
    recall_k = Nc / Ng
    F2_k = (5 * precision_k * recall_k) / (4 * precision_k + recall_k)

    F2_k[np.isnan(F2_k)] = 0

    ciwF2 = F2_k * weights
    ciwF2 = np.sum(ciwF2) / np.sum(weights)

    return ciwF2, F2_k


def evaluation(scores, targets, threshold=0.5):
    scores = scores[:, :, 1:]
    targets = targets[:, :, 1:]
    x, y, n_class = scores.shape
    scores = scores.reshape(x, n_class)
    targets = targets.reshape(x, n_class)
    assert scores.shape == targets.shape, "The input and targets do not have the same size: Input: {} - Targets: {}".format(
        scores.shape, targets.shape)
    #     print(444444444444444444444444444)
    #     print(scores.shape)
    # _, n_class = scores.shape
    #     print(5555555555555555555555555555)

    # Arrays to hold binary classification information, size n_class +1 to also hold the implicit normal class
    Nc = np.zeros(n_class + 1)  # Nc = Number of Correct Predictions  - True positives
    Np = np.zeros(n_class + 1)  # Np = Total number of Predictions    - True positives + False Positives
    Ng = np.zeros(n_class + 1)  # Ng = Total number of Ground Truth occurences

    # False Positives = Np - Nc
    # False Negatives = Ng - Nc
    # True Positives = Nc
    # True Negatives = n_examples - Np + (Ng - Nc)

    # Array to hold the average precision metric. only size n_class, since it is not possible to calculate for the implicit normal class
    ap = np.zeros(n_class)

    for k in range(n_class):
        tmp_scores = scores[:, k]
        tmp_targets = targets[:, k]
        tmp_targets[tmp_targets == -1] = 0  # Necessary if using MultiLabelSoftMarginLoss, instead of BCEWithLogitsLoss

        Ng[k] = np.sum(tmp_targets == 1)
        Np[k] = np.sum(tmp_scores >= threshold)  # when >= 0 for the raw input, the sigmoid value will be >= 0.5
        Nc[k] = np.sum(tmp_targets * (tmp_scores >= threshold))

        ap[k] = average_precision(tmp_scores, tmp_targets)

    # Get values for "implict" normal class
    tmp_scores = np.sum(scores >= threshold, axis=1)
    tmp_scores[tmp_scores > 0] = 1
    tmp_scores = np.abs(tmp_scores - 1)

    tmp_targets = targets.copy()
    tmp_targets[targets == -1] = 0  # Necessary if using MultiLabelSoftMarginLoss, instead of BCEWithLogitsLoss
    tmp_targets = np.sum(tmp_targets, axis=1)
    tmp_targets[tmp_targets > 0] = 1
    tmp_targets = np.abs(tmp_targets - 1)

    Ng[-1] = np.sum(tmp_targets == 1)
    Np[-1] = np.sum(tmp_scores >= threshold)
    Nc[-1] = np.sum(tmp_targets * (tmp_scores >= threshold))

    # If Np is 0 for any class, set to 1 to avoid division with 0
    Np[Np == 0] = 1

    # Overall Precision, Recall and F1
    OP, OR, OF1 = overall_metrics(Ng, Np, Nc)

    # Per-Class Precision, Recall and F1
    CP, CR, CF1 = per_class_metrics(Ng, Np, Nc)

    # Macro F1
    precision_k, recall_k, F1_k, MF1 = macro_f1(Ng, Np, Nc)

    # Micro F1
    mF1 = micro_f1(Ng, Np, Nc)

    # Zero-One exact match accuracy
    EMAcc = exact_match_accuracy(scores, targets)

    # Mean Average Precision (mAP)
    mAP = mean_average_precision(ap)

    # F2, F2_k, = class_weighted_f2(Ng[:-1], Np[:-1], Nc[:-1], weights)

    F2_normal = (5 * precision_k[-1] * recall_k[-1]) / (4 * precision_k[-1] + recall_k[-1])

    # new_metrics = {"F2": F2,
    #                "F2_class": list(F2_k) + [F2_normal],
    #                "F1_Normal": F1_k[-1]}
    new_metrics = {"F1_Normal": F1_k[-1]}

    main_metrics = {"OP": OP,
                    "OR": OR,
                    "OF1": OF1,
                    "CP": CP,
                    "CR": CR,
                    "CF1": CF1,
                    "MF1": MF1,
                    "mF1": mF1,
                    "EMAcc": EMAcc,
                    "mAP": mAP}

    auxillery_metrics = {"P_class": list(precision_k),
                         "R_class": list(recall_k),
                         "F1_class": list(F1_k),
                         "AP": list(ap),
                         "Np": list(Np),
                         "Nc": list(Nc),
                         "Ng": list(Ng)}

    return new_metrics, main_metrics, auxillery_metrics

def save_outputs(val_step_outputs, checkpoint):
    pred_list = val_step_outputs[::2]
    gt_list = val_step_outputs[1::2]
    pred_list = np.array(pred_list)
    gt_list = np.array(gt_list)
    tmp = checkpoint.split('/')
    tmp = tmp[-1].split('.')
    epoch = tmp[0]

    log_dir = 'C:/zcy/Video-Swin-Transformer-master/Video-Swin-Transformer-master/output/' + epoch
    view1_file = os.path.join(log_dir, 'view1_file.csv')
    if not os.path.isdir(log_dir):
        os.mkdir(log_dir)

    preds_view1 = pred_list
    gts_view1 = gt_list
    row1 = np.concatenate((preds_view1, gts_view1), axis=1)
    list1 = row1.tolist()

    firstrow = ['alpha', 'gt']
    with open(view1_file, 'a+', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(firstrow)
        writer.writerows(list1)

def validation_epoch_end(val_step_outputs, checkpoint):
    pred_list = val_step_outputs[::2]
    gt_list = val_step_outputs[1::2]
    pred_list = np.array(pred_list)
    gt_list = np.array(gt_list)
    tmp = checkpoint.split('/')
    epoch = tmp[-1]

    new_view1, main_view1, auxillary_view1 = evaluation(pred_list, gt_list)
    metrics_ret_view1 = [new_view1, main_view1, auxillary_view1]

    log_dir = r'C:\zcy\Video-Swin-Transformer-master\Video-Swin-Transformer-master\output'
    metrics_view1_file = os.path.join(log_dir, 'metrics_log.txt')

    with open(metrics_view1_file, 'a+') as f:
        save_string = epoch + '\n'
        for idx, result_dict in enumerate(metrics_ret_view1):
            for key in list(result_dict.keys()):
                save_string += '{}: {} '.format(key, result_dict[key])
                if idx == 2: save_string += '\n'
            save_string += '\n'
        f.write(save_string)

def turn_off_pretrained(cfg):
    # recursively find all pretrained in the model config,
    # and set them None to avoid redundant pretrain steps for testing
    if 'pretrained' in cfg:
        cfg.pretrained = None

    # recursively turn off pretrained value
    for sub_cfg in cfg.values():
        if isinstance(sub_cfg, dict):
            turn_off_pretrained(sub_cfg)


def inference_pytorch(args, cfg, distributed, data_loader, dataset, eval_config):
    """Get predictions by pytorch models."""
    if args.average_clips is not None:
        # You can set average_clips during testing, it will override the
        # original setting
        if cfg.model.get('test_cfg') is None and cfg.get('test_cfg') is None:
            cfg.model.setdefault('test_cfg',
                                 dict(average_clips=args.average_clips))
        else:
            if cfg.model.get('test_cfg') is not None:
                cfg.model.test_cfg.average_clips = args.average_clips
            else:
                cfg.test_cfg.average_clips = args.average_clips

    # remove redundant pretrain steps for testing
    turn_off_pretrained(cfg.model)

    # build the model and load checkpoint
    model = build_model(
        cfg.model, train_cfg=None, test_cfg=cfg.get('test_cfg'))

    if len(cfg.module_hooks) > 0:
        register_module_hooks(model, cfg.module_hooks)

    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is not None:
        wrap_fp16_model(model)

    # path = ''
    # input_template_All = []
    # f_list = os.listdir(args.checkpoint)  # 返回文件名
    # for i in f_list:
    #     # os.path.splitext():分离文件名与扩展名
    #     if os.path.splitext(i)[-1] == '.pth':
    #         input_template_All.append(i)
    #
    # for checkpoint in input_template_All:

    checkpoint_pth = args.checkpoint
    load_checkpoint(model, checkpoint_pth, map_location='cpu')

    if args.fuse_conv_bn:
        model = fuse_conv_bn(model)

    if not distributed:
        device_ids = cfg.gpu_ids
        model = MMDataParallel(model, device_ids=device_ids)
        outputs = single_gpu_test(model, data_loader)
    else:
        model = MMDistributedDataParallel(
            model.cuda(),
            device_ids=[torch.cuda.current_device()],
            broadcast_buffers=False)
        outputs = multi_gpu_test(model, data_loader, args.tmpdir,
                                 args.gpu_collect)

    return outputs


def inference_tensorrt(ckpt_path, distributed, data_loader, batch_size):
    """Get predictions by TensorRT engine.

    For now, multi-gpu mode and dynamic tensor shape are not supported.
    """
    assert not distributed, \
        'TensorRT engine inference only supports single gpu mode.'
    import tensorrt as trt
    from mmcv.tensorrt.tensorrt_utils import (torch_dtype_from_trt,
                                              torch_device_from_trt)

    # load engine
    with trt.Logger() as logger, trt.Runtime(logger) as runtime:
        with open(ckpt_path, mode='rb') as f:
            engine_bytes = f.read()
        engine = runtime.deserialize_cuda_engine(engine_bytes)

    # For now, only support fixed input tensor
    cur_batch_size = engine.get_binding_shape(0)[0]
    assert batch_size == cur_batch_size, \
        ('Dataset and TensorRT model should share the same batch size, '
         f'but get {batch_size} and {cur_batch_size}')

    context = engine.create_execution_context()

    # get output tensor
    dtype = torch_dtype_from_trt(engine.get_binding_dtype(1))
    shape = tuple(context.get_binding_shape(1))
    device = torch_device_from_trt(engine.get_location(1))
    output = torch.empty(
        size=shape, dtype=dtype, device=device, requires_grad=False)

    # get predictions
    results = []
    dataset = data_loader.dataset
    prog_bar = mmcv.ProgressBar(len(dataset))
    for data in data_loader:
        bindings = [
            data['imgs'].contiguous().data_ptr(),
            output.contiguous().data_ptr()
        ]
        context.execute_async_v2(bindings,
                                 torch.cuda.current_stream().cuda_stream)
        results.extend(output.cpu().numpy())
        batch_size = len(next(iter(data.values())))
        for _ in range(batch_size):
            prog_bar.update()
    return results


def inference_onnx(ckpt_path, distributed, data_loader, batch_size):
    """Get predictions by ONNX.

    For now, multi-gpu mode and dynamic tensor shape are not supported.
    """
    assert not distributed, 'ONNX inference only supports single gpu mode.'

    import onnx
    import onnxruntime as rt

    # get input tensor name
    onnx_model = onnx.load(ckpt_path)
    input_all = [node.name for node in onnx_model.graph.input]
    input_initializer = [node.name for node in onnx_model.graph.initializer]
    net_feed_input = list(set(input_all) - set(input_initializer))
    assert len(net_feed_input) == 1

    # For now, only support fixed tensor shape
    input_tensor = None
    for tensor in onnx_model.graph.input:
        if tensor.name == net_feed_input[0]:
            input_tensor = tensor
            break
    cur_batch_size = input_tensor.type.tensor_type.shape.dim[0].dim_value
    assert batch_size == cur_batch_size, \
        ('Dataset and ONNX model should share the same batch size, '
         f'but get {batch_size} and {cur_batch_size}')

    # get predictions
    sess = rt.InferenceSession(ckpt_path)
    results = []
    dataset = data_loader.dataset
    prog_bar = mmcv.ProgressBar(len(dataset))
    for data in data_loader:
        imgs = data['imgs'].cpu().numpy()
        onnx_result = sess.run(None, {net_feed_input[0]: imgs})[0]
        results.extend(onnx_result)
        batch_size = len(next(iter(data.values())))
        for _ in range(batch_size):
            prog_bar.update()
    return results


def main():
    args = parse_args()

    if args.tensorrt and args.onnx:
        raise ValueError(
            'Cannot set onnx mode and tensorrt mode at the same time.')

    cfg = Config.fromfile(args.config)

    cfg.merge_from_dict(args.cfg_options)

    # Load output_config from cfg
    output_config = cfg.get('output_config', {})
    if args.out:
        # Overwrite output_config from args.out
        output_config = Config._merge_a_into_b(
            dict(out=args.out), output_config)

    # Load eval_config from cfg
    eval_config = cfg.get('eval_config', {})
    if args.eval:
        # Overwrite eval_config from args.eval
        eval_config = Config._merge_a_into_b(
            dict(metrics=args.eval), eval_config)
    if args.eval_options:
        # Add options from args.eval_options
        eval_config = Config._merge_a_into_b(args.eval_options, eval_config)

    assert output_config or eval_config, \
        ('Please specify at least one operation (save or eval the '
         'results) with the argument "--out" or "--eval"')

    dataset_type = cfg.data.test.type
    if output_config.get('out', None):
        if 'output_format' in output_config:
            # ugly workround to make recognition and localization the same
            warnings.warn(
                'Skip checking `output_format` in localization task.')
        else:
            out = output_config['out']
            # make sure the dirname of the output path exists
            mmcv.mkdir_or_exist(osp.dirname(out))
            _, suffix = osp.splitext(out)
            if dataset_type == 'AVADataset':
                assert suffix[1:] == 'csv', ('For AVADataset, the format of '
                                             'the output file should be csv')
            else:
                assert suffix[1:] in file_handlers, (
                    'The format of the output '
                    'file should be json, pickle or yaml')

    # set cudnn benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True
    cfg.data.test.test_mode = True

    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)

    # The flag is used to register module's hooks
    cfg.setdefault('module_hooks', [])

    # build the dataloader
    dataset = build_dataset(cfg.data.test, dict(test_mode=True))
    dataloader_setting = dict(
        videos_per_gpu=cfg.data.get('videos_per_gpu', 1),
        workers_per_gpu=cfg.data.get('workers_per_gpu', 1),
        dist=distributed,
        shuffle=False)
    dataloader_setting = dict(dataloader_setting,
                              **cfg.data.get('test_dataloader', {}))
    data_loader = build_dataloader(dataset, **dataloader_setting)

    if args.tensorrt:
        outputs = inference_tensorrt(args.checkpoint, distributed, data_loader,
                                     dataloader_setting['videos_per_gpu'])
    elif args.onnx:
        outputs = inference_onnx(args.checkpoint, distributed, data_loader,
                                 dataloader_setting['videos_per_gpu'])
    else:
        outputs = inference_pytorch(args, cfg, distributed, data_loader, dataset, eval_config)

    save_outputs(outputs, args.checkpoint)
    validation_epoch_end(outputs, args.checkpoint)
    # f = open('C:/zcy/Video-Swin-Transformer-master/Video-Swin-Transformer-master/output/test_log.txt', 'a+')
    # sys.stdout = f
    # rank, _ = get_dist_info()
    # print(str(args.checkpoint))
    # if rank == 0:
    #     if output_config.get('out', None):
    #         out = output_config['out']
    #         print(f'\nwriting results to {out}')
    #         dataset.dump_results(outputs, **output_config)
    #     if eval_config:
    #         eval_res = dataset.evaluate(outputs, **eval_config)
    #         for name, val in eval_res.items():
    #             print(f'{name}: {val:.04f}')
    # f.close()

if __name__ == '__main__':
    main()
