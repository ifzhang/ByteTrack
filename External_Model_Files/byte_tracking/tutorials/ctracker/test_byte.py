import numpy as np
import torchvision
import time
import math
import os
import copy
import pdb
import argparse
import sys
import cv2
import skimage.io
import skimage.transform
import skimage.color
import skimage
import torch
import model

from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, models, transforms
from dataloader import CSVDataset, collater, Resizer, AspectRatioBasedSampler, Augmenter, UnNormalizer, Normalizer, RGB_MEAN, RGB_STD
from scipy.optimize import linear_sum_assignment
from tracker import BYTETracker


def write_results(filename, results):
    save_format = '{frame},{id},{x1},{y1},{w},{h},{s},-1,-1,-1\n'
    with open(filename, 'w') as f:
        for frame_id, tlwhs, track_ids, scores in results:
            for tlwh, track_id, score in zip(tlwhs, track_ids, scores):
                if track_id < 0:
                    continue
                x1, y1, w, h = tlwh
                line = save_format.format(frame=frame_id, id=track_id, x1=round(x1, 1), y1=round(y1, 1), w=round(w, 1), h=round(h, 1), s=round(score, 2))
                f.write(line)

def write_results_no_score(filename, results):
    save_format = '{frame},{id},{x1},{y1},{w},{h},-1,-1,-1,-1\n'
    with open(filename, 'w') as f:
        for frame_id, tlwhs, track_ids in results:
            for tlwh, track_id in zip(tlwhs, track_ids):
                if track_id < 0:
                    continue
                x1, y1, w, h = tlwh
                line = save_format.format(frame=frame_id, id=track_id, x1=round(x1, 1), y1=round(y1, 1), w=round(w, 1), h=round(h, 1))
                f.write(line)
                
def run_each_dataset(model_dir, retinanet, dataset_path, subset, cur_dataset):
    print(cur_dataset)

    img_list = os.listdir(os.path.join(dataset_path, subset, cur_dataset, 'img1'))
    img_list = [os.path.join(dataset_path, subset, cur_dataset, 'img1', _) for _ in img_list if ('jpg' in _) or ('png' in _)]
    img_list = sorted(img_list)

    img_len = len(img_list)
    last_feat = None

    confidence_threshold = 0.6
    IOU_threshold = 0.5
    retention_threshold = 10

    det_list_all = []
    tracklet_all = []
    results = []
    max_id = 0
    max_draw_len = 100
    draw_interval = 5
    img_width = 1920
    img_height = 1080
    fps = 30

    tracker = BYTETracker()

    for idx in range((int(img_len / 2)), img_len + 1):
        i = idx - 1
        print('tracking: ', i)
        with torch.no_grad():
            data_path1 = img_list[min(idx, img_len - 1)]
            img_origin1 = skimage.io.imread(data_path1)
            img_h, img_w, _ = img_origin1.shape
            img_height, img_width = img_h, img_w
            resize_h, resize_w = math.ceil(img_h / 32) * 32, math.ceil(img_w / 32) * 32
            img1 = np.zeros((resize_h, resize_w, 3), dtype=img_origin1.dtype)
            img1[:img_h, :img_w, :] = img_origin1
            img1 = (img1.astype(np.float32) / 255.0 - np.array([[RGB_MEAN]])) / np.array([[RGB_STD]])
            img1 = torch.from_numpy(img1).permute(2, 0, 1).view(1, 3, resize_h, resize_w)
            scores, transformed_anchors, last_feat = retinanet(img1.cuda().float(), last_feat=last_feat)
            
            if idx > (int(img_len / 2)):
                idxs = np.where(scores > 0.1)
                # run tracking
                online_targets = tracker.update(transformed_anchors[idxs[0], :4], scores[idxs[0]])
                online_tlwhs = []
                online_ids = []
                online_scores = []
                for t in online_targets:
                    tlwh = t.tlwh
                    tid = t.track_id
                    online_tlwhs.append(tlwh)
                    online_ids.append(tid)
                    online_scores.append(t.score)
                results.append((idx, online_tlwhs, online_ids, online_scores))
            
    fout_tracking = os.path.join(model_dir, 'results', cur_dataset + '.txt')
    write_results(fout_tracking, results)           

    

def main(args=None):
    parser = argparse.ArgumentParser(description='Simple script for testing a CTracker network.')
    parser.add_argument('--dataset_path', default='/dockerdata/home/jeromepeng/data/MOT/MOT17/', type=str,
                        help='Dataset path, location of the images sequence.')
    parser.add_argument('--model_dir', default='./trained_model/', help='Path to model (.pt) file.')
    parser.add_argument('--model_path', default='./trained_model/model_final.pth', help='Path to model (.pt) file.')
    parser.add_argument('--seq_nums', default=0, type=int)

    parser = parser.parse_args(args)

    if not os.path.exists(os.path.join(parser.model_dir, 'results')):
        os.makedirs(os.path.join(parser.model_dir, 'results'))

    retinanet = model.resnet50(num_classes=1, pretrained=True)
    # 	retinanet_save = torch.load(os.path.join(parser.model_dir, 'model_final.pth'))
    retinanet_save = torch.load(os.path.join(parser.model_path))

    # rename moco pre-trained keys
    state_dict = retinanet_save.state_dict()
    for k in list(state_dict.keys()):
        # retain only encoder up to before the embedding layer
        if k.startswith('module.'):
            # remove prefix
            state_dict[k[len("module."):]] = state_dict[k]
        # delete renamed or unused k
        del state_dict[k]

    retinanet.load_state_dict(state_dict)

    use_gpu = True

    if use_gpu: retinanet = retinanet.cuda()

    retinanet.eval()
    seq_nums = []
    if parser.seq_nums > 0:
        seq_nums.append(parser.seq_nums)
    else:
        seq_nums = [2, 4, 5, 9, 10, 11, 13]

    for seq_num in seq_nums:
        run_each_dataset(parser.model_dir, retinanet, parser.dataset_path, 'train', 'MOT17-{:02d}'.format(seq_num))


# 	for seq_num in [1, 3, 6, 7, 8, 12, 14]:
# 		run_each_dataset(parser.model_dir, retinanet, parser.dataset_path, 'test', 'MOT17-{:02d}'.format(seq_num))

if __name__ == '__main__':
    main()
