import os.path as osp

from .base import BaseDataset
from .builder import DATASETS


@DATASETS.register_module()
class Multiview_OOD(BaseDataset):
    """Video dataset for action recognition.

    The dataset loads raw videos and apply specified transforms to return a
    dict containing the frame tensors and other information.

    The ann_file is a text file with multiple lines, and each line indicates
    a sample video with the filepath and label, which are split with a
    whitespace. Example of a annotation file:

    .. code-block:: txt

        some/path/000.mp4 1
        some/path/001.mp4 1
        some/path/002.mp4 2
        some/path/003.mp4 2
        some/path/004.mp4 3
        some/path/005.mp4 3


    Args:
        ann_file (str): Path to the annotation file.
        pipeline (list[dict | callable]): A sequence of data transforms.
        start_index (int): Specify a start index for frames in consideration of
            different filename format. However, when taking videos as input,
            it should be set to 0, since frames loaded from videos count
            from 0. Default: 0.
        **kwargs: Keyword arguments for ``BaseDataset``.
    """

    def __init__(self, ann_file, pipeline, start_index=0, **kwargs):
        super().__init__(ann_file, pipeline, start_index=start_index, **kwargs)


    def load_annotations(self):
        """Load annotation file to get video information."""
        if self.ann_file.endswith('.json'):
            return self.load_json_annotations()

        pth = r'C:\Video-Swin-Transformer-master\data\multi_view_result.txt'
        lines = open(pth).readlines()
        lines = [x.split('\t') for x in lines]

        video_infos = []
        with open(self.ann_file, 'r') as fin:
            for line in fin:
                line_split = line.strip().split()
                # if self.multi_class:
                #     assert self.num_classes is not None
                if line_split[2:-1] == []:
                    ood = line_split[0]
                    ood = list(map(int, ood))
                    filename, label, total_frames = line_split[1], line_split[2:-1], line_split[-1]
                    label = list(map(int, label))
                    lines.sort(key=lambda x: x[0] != filename)
                    oneline = lines[0]
                    index_out1 = oneline.index(' out1:')
                    index_out2 = oneline.index(' out2:')
                    per_view1 = oneline[index_out1 + 1: index_out2]
                    per_view2 = oneline[index_out2 + 1: -1]
                else:
                    ood = line_split[0]
                    ood = list(map(int, ood))
                    filename, label, total_frames = line_split[1], line_split[2:-1], line_split[-1]
                    label = list(map(int, label))
                    lines.sort(key=lambda x: x[0] != filename)
                    oneline = lines[0]
                    index_out1 = oneline.index(' out1:')
                    index_out2 = oneline.index(' out2:')
                    per_view1 = oneline[index_out1 + 1: index_out2]
                    per_view2 = oneline[index_out2 + 1: -1]
                # else:
                #     filename, label = line_split
                #     label = int(label)
                if self.data_prefix is not None:
                    filename = osp.join(self.data_prefix, filename)
                frame_dir = 'C:/Video-Swin-Transformer-master/data/track1/track1_raw_video'
                video_infos.append(dict(filename=filename, label=label, per_view1=per_view1, per_view2=per_view2,
                                        OOD_label=ood))
        return video_infos
