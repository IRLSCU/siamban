from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os
import argparse

import cv2
import torch
import numpy as np
from glob import glob

from siamban.core.config import cfg
from siamban.models.model_builder import ModelBuilder
from siamban.tracker.tracker_builder import build_tracker
from siamban.utils.model_load import load_pretrain

torch.set_num_threads(1)


def get_frames(video_name):
    """
    数据帧获取函数
    Args:
        video_name ([type]): 视频名称

    Yields:
        [img]: opencv视频数据
    """

    if not video_name:
        cap = cv2.VideoCapture(0)
        # warmup
        # 先读取5帧图像
        for i in range(5):
            cap.read()
        while True:
            ret, frame = cap.read()
            if ret:
                yield frame
            else:
                break
    # 根据文件名，判断视频文件
    elif video_name.endswith('avi') or \
        video_name.endswith('mp4') or \
        video_name.endswith('mov'):
        # 创建视频摄像头
        cap = cv2.VideoCapture(args.video_name)
        while True:
            ret, frame = cap.read()
            if ret:
                yield frame
            else:
                break
    else:
        images = glob(os.path.join(video_name, '*.jp*'))
        images = sorted(images,
                        key=lambda x: int(x.split('/')[-1].split('.')[0]))
        for img in images:
            frame = cv2.imread(img)
            yield frame


def main(args):
    # load config
    cfg.merge_from_file(args.config)
    cfg.CUDA = torch.cuda.is_available() and cfg.CUDA
    device = torch.device('cuda' if cfg.CUDA else 'cpu')

    # create model
    model = ModelBuilder()

    # load model
    # model.load_state_dict(torch.load(args.snapshot,
    #     map_location=lambda storage, loc: storage.cpu()))
    # model.eval().to(device)

    # load model
    model = load_pretrain(model, args.snapshot).cuda().eval()

    # build tracker
    # 创建追踪器
    tracker = build_tracker(model)

    first_frame = True
    if args.video_name:
        video_name = args.video_name.split('/')[-1].split('.')[0]
    else:
        video_name = 'webcam'
    cv2.namedWindow(video_name, cv2.WND_PROP_FULLSCREEN)
        
    for frame in get_frames(args.video_name):
        # 对于第一帧需要进行初始化
        if first_frame:
            # build video writer
            # 构建视频写入者
            if args.save:
                if args.video_name.endswith('avi') or \
                    args.video_name.endswith('mp4') or \
                    args.video_name.endswith('mov'):
                    cap = cv2.VideoCapture(args.video_name)
                    fps = int(round(cap.get(cv2.CAP_PROP_FPS)))
                else:
                    fps = 30
                # 设置存储视频名称
                save_video_path = args.video_name.split(video_name)[0] + video_name + '_tracking.mp4'
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                frame_size = (frame.shape[1], frame.shape[0]) # (w, h)
                video_writer = cv2.VideoWriter(save_video_path, fourcc, fps, frame_size)
            try:
                # 获取视频输入,获取输入的图像框
                init_rect = cv2.selectROI(video_name, frame, False, False)
            except:
                exit()
            # 初始化追踪器
            tracker.init(frame, init_rect)
            first_frame = False
        else:
            # 进行模型推理
            outputs = tracker.track(frame)
            # 如果输出中存在多边型
            if 'polygon' in outputs:
                polygon = np.array(outputs['polygon']).astype(np.int32)
                cv2.polylines(frame, [polygon.reshape((-1, 1, 2))],
                              True, (0, 255, 0), 3)
                mask = ((outputs['mask'] > cfg.TRACK.MASK_THERSHOLD) * 255)
                mask = mask.astype(np.uint8)
                mask = np.stack([mask, mask*255, mask]).transpose(1, 2, 0)
                frame = cv2.addWeighted(frame, 0.77, mask, 0.23, -1)
            else:
                # 获取最终输出最表，
                bbox = list(map(int, outputs['bbox']))
                '''
                    矩形框的选取
                    x1,y1 ------
                    |          |
                    |          |
                    |          |
                    --------x2,y2
                '''
                cv2.rectangle(frame, (bbox[0], bbox[1]),
                              (bbox[0]+bbox[2], bbox[1]+bbox[3]),
                              (0, 255, 0), 3)
            cv2.imshow(video_name, frame)
            cv2.waitKey(40)

        if args.save:
            video_writer.write(frame)
    
    if args.save:
        video_writer.release()


if __name__ == '__main__':
    """
    主函数
    """
    parser = argparse.ArgumentParser(description='tracking demo')
    # 配置文件参数
    parser.add_argument('--config', type=str, help='config file')
    # 模型名称
    parser.add_argument('--snapshot', type=str, help='model name')
    # 视频名称
    parser.add_argument('--video_name', default='', type=str,
                        help='videos or image files')
    parser.add_argument('--save', action='store_true',help='whether visualzie result')
    args = parser.parse_args()
    main(args)
