# -*- coding: utf-8 -*-
"""
-------------------------------------------------
# @Project  :SuperGlue_training
# @File     :demo
# @Date     :2024/8/19 19:56
# @Author   :SunRui
# @Software :PyCharm
-------------------------------------------------
"""

from pathlib import Path
import argparse
import cv2
import matplotlib.cm as cm
import torch
import os
from models.matching import Matching
from utils.common import (AverageTimer, VideoStreamer,
                          make_matching_plot_fast, frame2tensor,download_base_files, weights_mapping)

torch.set_grad_enabled(False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='SuperGlue demo',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '--input', type=str, default='0',
        help='ID of a USB webcam, URL of an IP camera, '
             'or path to an images directory or movie file')
    parser.add_argument(
        '--output_dir', type=str, default=None,
        help='Directory where to write output frames (If None, no output)')

    parser.add_argument(
        '--image_glob', type=str, nargs='+', default=['*.png', '*.jpg', '*.jpeg'],
        help='Glob if a directory of images is specified')
    parser.add_argument(
        '--skip', type=int, default=1,
        help='Images to skip if input is a movie or directory')
    parser.add_argument(
        '--max_length', type=int, default=1000000,
        help='Maximum length if input is a movie or directory')
    parser.add_argument(
        '--resize', type=int, nargs='+', default=[640, 480],
        help='Resize the input images before running inference. If two numbers, '
             'resize to the exact dimensions, if one number, resize the max '
             'dimension, if -1, do not resize')

    parser.add_argument(
        '--superglue', default='outdoor',#coco_homo     outdoor
        help='SuperGlue weights')
    parser.add_argument(
        '--max_keypoints', type=int, default=-1,
        help='Maximum number of keypoints detected by Superpoint'
             ' (\'-1\' keeps all keypoints)')
    parser.add_argument(
        '--keypoint_threshold', type=float, default=0.05,
        help='SuperPoint keypoint detector confidence threshold')
    parser.add_argument(
        '--nms_radius', type=int, default=4,
        help='SuperPoint Non Maximum Suppression (NMS) radius'
        ' (Must be positive)')
    parser.add_argument(
        '--sinkhorn_iterations', type=int, default=20,
        help='Number of Sinkhorn iterations performed by SuperGlue')
    parser.add_argument(
        '--match_threshold', type=float, default=0.2,
        help='SuperGlue match threshold')

    parser.add_argument(
        '--show_keypoints', action='store_true',
        help='Show the detected keypoints')
    parser.add_argument(
        '--no_display', action='store_true',
        help='Do not display images to screen. Useful if running remotely')
    parser.add_argument(
        '--force_cpu', action='store_true',
        help='Force pytorch to run in CPU mode.')

    opt = parser.parse_args()
    print(opt)

    if len(opt.resize) == 2 and opt.resize[1] == -1:
        opt.resize = opt.resize[0:1]
    if len(opt.resize) == 2:
        print('Will resize to {}x{} (WxH)'.format(
            opt.resize[0], opt.resize[1]))
    elif len(opt.resize) == 1 and opt.resize[0] > 0:
        print('Will resize max dimension to {}'.format(opt.resize[0]))
    elif len(opt.resize) == 1:
        print('Will not resize images')
    else:
        raise ValueError('Cannot specify more than two integers for --resize')
    # download_base_files()
    device = 'cuda' if torch.cuda.is_available() and not opt.force_cpu else 'cpu'
    print('Running inference on device \"{}\"'.format(device))
    try:
        curr_weights_path = str(weights_mapping[opt.superglue])
    except:
        if os.path.isfile(opt.superglue) and (os.path.splitext(opt.superglue)[-1] in ['.pt', '.pth']):
            curr_weights_path = str(opt.superglue)
        else:
            raise ValueError("Given --superglue path doesn't exist or invalid")
    config = {
        'superpoint': {
            'nms_radius': opt.nms_radius,
            'keypoint_threshold': opt.keypoint_threshold,
            'max_keypoints': opt.max_keypoints
        },
        'superglue': {
            'weights_path': curr_weights_path,
            'sinkhorn_iterations': opt.sinkhorn_iterations,
            'match_threshold': opt.match_threshold,
        }
    }
    matching = Matching(config).eval().to(device)
    keys = ['keypoints', 'scores', 'descriptors']

    image0 = cv2.imread('images/image1.jpg')[:, :, 0]
    image1 = cv2.imread('images/image2.jpg')[:,:,0]
    if image0 is None or image1 is None: raise ValueError(f"image could not be read,is none")

    image0_tensor = frame2tensor(image0, device)
    image1_tensor = frame2tensor(image1, device)
    image0_ksd = matching.superpoint({'images': image0_tensor})
    image1_ksd = matching.superpoint({'images': image1_tensor})

    image0_ksd = {k + '0': image0_ksd[k] for k in keys} #对键值重命名
    image1_ksd = {k + '1': image1_ksd[k] for k in keys}
    image0_ksd['image0'] = image0_tensor #将image1的tensor数据保存在键值image0中
    image1_ksd['image0'] = image1_tensor

    image0_source = image0 #保留原始数据
    image1_source = image1

    output_dir = "./output"


    # Create a window to display the demo.
    if not opt.no_display:
        cv2.namedWindow('SuperGlue matches', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('SuperGlue matches', (640*2, 480))
    else:
        print('Skipping visualization, will not show a GUI.')

    timer = AverageTimer()


    pred = matching({**image0_ksd, 'image1': image1_tensor})
    kpts0 = image0_ksd['keypoints0'][0].cpu().numpy()
    kpts1 = pred['keypoints1'][0].cpu().numpy()
    matches = pred['matches0'][0].cpu().numpy()
    confidence = pred['matching_scores0'][0].cpu().numpy()

    valid = matches > -1
    mkpts0 = kpts0[valid]
    mkpts1 = kpts1[matches[valid]]
    color = cm.jet(confidence[valid])

    text = [
        'SuperGlue',
        'Keypoints: {}:{}'.format(len(kpts0), len(kpts1)),
        'Matches: {}'.format(len(mkpts0))
    ]
    k_thresh = matching.superpoint.config['keypoint_threshold']
    m_thresh = matching.superglue.config['match_threshold']
    small_text = [
        'Keypoint Threshold: {:.4f}'.format(k_thresh),
        'Match Threshold: {:.2f}'.format(m_thresh),
        'Image Pair: {}:{}'.format("image0", "image1"),
    ]
    out = make_matching_plot_fast(
        image0_source, image1_source, kpts0, kpts1, mkpts0, mkpts1, color, text,
        path=None, show_keypoints=opt.show_keypoints, small_text=small_text)

    cv2.imshow('SuperGlue matches', out)
    key = chr(cv2.waitKey(1) & 0xFF)
    # 修改为等待任意键按下，不设置超时
    cv2.waitKey(0)
    # 在用户按下键后关闭所有OpenCV窗口
    cv2.destroyAllWindows()










# import cv2
# from matplotlib import cm
#
# from models.matching import Matching
# from utils.common import (frame2tensor, make_matching_plot_fast)
#
# # Load the images directly using OpenCV
# image0 = cv2.imread('images/image1.jpg')
# image1 = cv2.imread('images/image2.jpg')
#
# # Resize if necessary
# resize = [640, 480]
# image0 = cv2.resize(image0, (resize[0], resize[1]))
# image1 = cv2.resize(image1, (resize[0], resize[1]))
#
# # Convert images to tensor
# frame_tensor0 = frame2tensor(image0, 'cuda')
# frame_tensor1 = frame2tensor(image1, 'cuda')
#
# # Matching process
# config = {
#     'superpoint': {
#         'nms_radius': 4,
#         'keypoint_threshold': 0.005,
#         'max_keypoints': -1
#     },
#     'superglue': {
#         'weights_path': 'models/weights/superglue_cocohomo.pt',
#         'sinkhorn_iterations': 20,
#         'match_threshold': 0.2,
#     }
# }
# matching = Matching(config).eval().to('cuda')
# last_data = matching.superpoint({'images ': frame_tensor0})
# last_data = {k+'0': last_data[k] for k in ['keypoints', 'scores', 'descriptors']}
# last_data['image0'] = frame_tensor0
#
# pred = matching({**last_data, 'image1': frame_tensor1})
#
# # Process and visualize the matches
# kpts0 = last_data['keypoints0'][0].cpu().numpy()
# kpts1 = pred['keypoints1'][0].cpu().numpy()
# matches = pred['matches0'][0].cpu().numpy()
# confidence = pred['matching_scores0'][0].cpu().numpy()
#
# valid = matches > -1
# mkpts0 = kpts0[valid]
# mkpts1 = kpts1[matches[valid]]
# color = cm.jet(confidence[valid])
#
# out = make_matching_plot_fast(
#     image0, image1, kpts0, kpts1, mkpts0, mkpts1, color, text=['SuperGlue'],
#     show_keypoints=False, path=None)
#
# cv2.imwrite('output/matches.png', out)

