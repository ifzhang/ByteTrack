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

# assert torch.__version__.split('.')[1] == '4'

print('CUDA available: {}'.format(torch.cuda.is_available()))

color_list = [(0, 0, 255), (255, 0, 0), (0, 255, 0), (255, 0, 255), (0, 255, 255), (255, 255, 0), (128, 0, 255), 
(0, 128, 255), (128, 255, 0), (0, 255, 128), (255, 128, 0), (255, 0, 128), (128, 128, 255), (128, 255, 128), (255, 128, 128), (128, 128, 0), (128, 0, 128)]

class detect_rect:
	def __init__(self):
		self.curr_frame = 0
		self.curr_rect = np.array([0, 0, 1, 1])
		self.next_rect = np.array([0, 0, 1, 1])
		self.conf = 0
		self.id = 0

	@property
	def position(self):
		x = (self.curr_rect[0] + self.curr_rect[2])/2
		y = (self.curr_rect[1] + self.curr_rect[3])/2
		return np.array([x, y])

	@property
	def size(self):
		w = self.curr_rect[2] - self.curr_rect[0]
		h = self.curr_rect[3] - self.curr_rect[1]
		return np.array([w, h])

class tracklet:
	def __init__(self, det_rect):
		self.id = det_rect.id
		self.rect_list = [det_rect]
		self.rect_num = 1
		self.last_rect = det_rect
		self.last_frame = det_rect.curr_frame
		self.no_match_frame = 0

	def add_rect(self, det_rect):
		self.rect_list.append(det_rect)
		self.rect_num = self.rect_num + 1
		self.last_rect = det_rect
		self.last_frame = det_rect.curr_frame

	@property
	def velocity(self):
		if(self.rect_num < 2):
			return (0, 0)
		elif(self.rect_num < 6):
			return (self.rect_list[self.rect_num - 1].position - self.rect_list[self.rect_num - 2].position) / (self.rect_list[self.rect_num - 1].curr_frame - self.rect_list[self.rect_num - 2].curr_frame)
		else:
			v1 = (self.rect_list[self.rect_num - 1].position - self.rect_list[self.rect_num - 4].position) / (self.rect_list[self.rect_num - 1].curr_frame - self.rect_list[self.rect_num - 4].curr_frame)
			v2 = (self.rect_list[self.rect_num - 2].position - self.rect_list[self.rect_num - 5].position) / (self.rect_list[self.rect_num - 2].curr_frame - self.rect_list[self.rect_num - 5].curr_frame)
			v3 = (self.rect_list[self.rect_num - 3].position - self.rect_list[self.rect_num - 6].position) / (self.rect_list[self.rect_num - 3].curr_frame - self.rect_list[self.rect_num - 6].curr_frame)
			return (v1 + v2 + v3) / 3


def cal_iou(rect1, rect2):
	x1, y1, x2, y2 = rect1
	x3, y3, x4, y4 = rect2
	i_w = min(x2, x4) - max(x1, x3)
	i_h = min(y2, y4) - max(y1, y3)
	if(i_w <= 0 or i_h <= 0):
		return 0
	i_s = i_w * i_h
	s_1 = (x2 - x1) * (y2 - y1)
	s_2 = (x4 - x3) * (y4 - y3)
	return float(i_s) / (s_1 + s_2 - i_s) 

def cal_simi(det_rect1, det_rect2):
	return cal_iou(det_rect1.next_rect, det_rect2.curr_rect)

def cal_simi_track_det(track, det_rect):
	if(det_rect.curr_frame <= track.last_frame):
		print("cal_simi_track_det error")
		return 0
	elif(det_rect.curr_frame - track.last_frame == 1):
		return cal_iou(track.last_rect.next_rect, det_rect.curr_rect)
	else:
		pred_rect = track.last_rect.curr_rect + np.append(track.velocity, track.velocity) * (det_rect.curr_frame - track.last_frame)
		return cal_iou(pred_rect, det_rect.curr_rect)

def track_det_match(tracklet_list, det_rect_list, min_iou = 0.5):
	num1 = len(tracklet_list)
	num2 = len(det_rect_list)
	cost_mat = np.zeros((num1, num2))
	for i in range(num1):
		for j in range(num2):
			cost_mat[i, j] = -cal_simi_track_det(tracklet_list[i], det_rect_list[j])

	match_result = linear_sum_assignment(cost_mat)
	match_result = np.asarray(match_result)
	match_result = np.transpose(match_result)

	matches, unmatched1, unmatched2 = [], [], []
	for i in range(num1):
		if i not in match_result[:, 0]:
			unmatched1.append(i)
	for j in range(num2):
		if j not in match_result[:, 1]:
			unmatched2.append(j)
	for i, j in match_result:
		if cost_mat[i, j] > -min_iou:
			unmatched1.append(i)
			unmatched2.append(j)
		else:
			matches.append((i, j))
	return matches, unmatched1, unmatched2

def draw_caption(image, box, caption, color):
	b = np.array(box).astype(int)
	cv2.putText(image, caption, (b[0], b[1] - 8), cv2.FONT_HERSHEY_PLAIN, 2, color, 2)


def run_each_dataset(model_dir, retinanet, dataset_path, subset, cur_dataset):	
	print(cur_dataset)

	img_list = os.listdir(os.path.join(dataset_path, subset, cur_dataset, 'img1'))
	img_list = [os.path.join(dataset_path, subset, cur_dataset, 'img1', _) for _ in img_list if ('jpg' in _) or ('png' in _)]
	img_list = sorted(img_list)

	img_len = len(img_list)
	last_feat = None

	confidence_threshold = 0.4
	IOU_threshold = 0.5
	retention_threshold = 10

	det_list_all = []
	tracklet_all = []
	max_id = 0
	max_draw_len = 100
	draw_interval = 5
	img_width = 1920
	img_height = 1080
	fps = 30

	for i in range(img_len):
		det_list_all.append([])

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
# 			if idx > 0:
			if idx > (int(img_len / 2)):
				idxs = np.where(scores>0.1)

				for j in range(idxs[0].shape[0]):
					bbox = transformed_anchors[idxs[0][j], :]
					x1 = int(bbox[0])
					y1 = int(bbox[1])
					x2 = int(bbox[2])
					y2 = int(bbox[3])

					x3 = int(bbox[4])
					y3 = int(bbox[5])
					x4 = int(bbox[6])
					y4 = int(bbox[7])

					det_conf = float(scores[idxs[0][j]])

					det_rect = detect_rect()
					det_rect.curr_frame = idx
					det_rect.curr_rect = np.array([x1, y1, x2, y2])
					det_rect.next_rect = np.array([x3, y3, x4, y4])
					det_rect.conf = det_conf

					if det_rect.conf > confidence_threshold:
						det_list_all[det_rect.curr_frame - 1].append(det_rect)
# 				if i == 0:
				if i == int(img_len / 2):
					for j in range(len(det_list_all[i])):
						det_list_all[i][j].id = j + 1
						max_id = max(max_id, j + 1)
						track = tracklet(det_list_all[i][j])
						tracklet_all.append(track)
					continue

				matches, unmatched1, unmatched2 = track_det_match(tracklet_all, det_list_all[i], IOU_threshold)

				for j in range(len(matches)):
					det_list_all[i][matches[j][1]].id = tracklet_all[matches[j][0]].id
					det_list_all[i][matches[j][1]].id = tracklet_all[matches[j][0]].id
					tracklet_all[matches[j][0]].add_rect(det_list_all[i][matches[j][1]])

				delete_track_list = []
				for j in range(len(unmatched1)):
					tracklet_all[unmatched1[j]].no_match_frame = tracklet_all[unmatched1[j]].no_match_frame + 1
					if(tracklet_all[unmatched1[j]].no_match_frame >= retention_threshold):
						delete_track_list.append(unmatched1[j])

				origin_index = set([k for k in range(len(tracklet_all))])
				delete_index = set(delete_track_list)
				left_index = list(origin_index - delete_index)
				tracklet_all = [tracklet_all[k] for k in left_index]


				for j in range(len(unmatched2)):
					det_list_all[i][unmatched2[j]].id = max_id + 1
					max_id = max_id + 1
					track = tracklet(det_list_all[i][unmatched2[j]])
					tracklet_all.append(track)

                    

	#**************visualize tracking result and save evaluate file****************

	fout_tracking = open(os.path.join(model_dir, 'results', cur_dataset + '.txt'), 'w')

	save_img_dir = os.path.join(model_dir, 'results', cur_dataset)
	if not os.path.exists(save_img_dir):
		os.makedirs(save_img_dir)

	out_video = os.path.join(model_dir, 'results', cur_dataset + '.mp4')
	videoWriter = cv2.VideoWriter(out_video, cv2.VideoWriter_fourcc('m', 'p', '4', 'v'), fps, (img_width, img_height))

	id_dict = {}


	for i in range((int(img_len / 2)), img_len):
		print('saving: ', i)
		img = cv2.imread(img_list[i])

		for j in range(len(det_list_all[i])):

			x1, y1, x2, y2 = det_list_all[i][j].curr_rect.astype(int)
			trace_id = det_list_all[i][j].id

			id_dict.setdefault(str(trace_id),[]).append((int((x1+x2)/2), y2))
			draw_trace_id = str(trace_id)
			draw_caption(img, (x1, y1, x2, y2), draw_trace_id, color=color_list[trace_id % len(color_list)])
			cv2.rectangle(img, (x1, y1), (x2, y2), color=color_list[trace_id % len(color_list)], thickness=2)

			trace_len = len(id_dict[str(trace_id)])
			trace_len_draw = min(max_draw_len, trace_len)
			
			for k in range(trace_len_draw - draw_interval):
				if(k % draw_interval == 0):
					draw_point1 = id_dict[str(trace_id)][trace_len - k - 1]
					draw_point2 = id_dict[str(trace_id)][trace_len - k - 1 - draw_interval]
					cv2.line(img, draw_point1, draw_point2, color=color_list[trace_id % len(color_list)], thickness=2)

			fout_tracking.write(str(i+1) + ',' + str(trace_id) + ',' + str(x1) + ',' + str(y1) + ',' + str(x2 - x1) + ',' + str(y2 - y1) + ',-1,-1,-1,-1\n')

		cv2.imwrite(os.path.join(save_img_dir, str(i + 1).zfill(6) + '.jpg'), img)
		videoWriter.write(img)
# 		cv2.waitKey(0)

	fout_tracking.close()
	videoWriter.release()

def run_from_train(model_dir, root_path):
	if not os.path.exists(os.path.join(model_dir, 'results')):
		os.makedirs(os.path.join(model_dir, 'results'))
	retinanet = torch.load(os.path.join(model_dir, 'model_final.pt'))

	use_gpu = True

	if use_gpu: retinanet = retinanet.cuda()

	retinanet.eval()

	for seq_num in [2, 4, 5, 9, 10, 11, 13]:
		run_each_dataset(model_dir, retinanet, root_path, 'train', 'MOT17-{:02d}'.format(seq_num))
	for seq_num in [1, 3, 6, 7, 8, 12, 14]:
		run_each_dataset(model_dir, retinanet, root_path, 'test', 'MOT17-{:02d}'.format(seq_num))

def main(args=None):
	parser = argparse.ArgumentParser(description='Simple script for testing a CTracker network.')
	parser.add_argument('--dataset_path', default='/dockerdata/home/jeromepeng/data/MOT/MOT17/', type=str, help='Dataset path, location of the images sequence.')
	parser.add_argument('--model_dir', default='./trained_model/', help='Path to model (.pt) file.')
	parser.add_argument('--model_path', default='./trained_model/model_final.pth', help='Path to model (.pt) file.')
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

	for seq_num in [2, 4, 5, 9, 10, 11, 13]:
		run_each_dataset(parser.model_dir, retinanet, parser.dataset_path, 'train', 'MOT17-{:02d}'.format(seq_num))
# 	for seq_num in [1, 3, 6, 7, 8, 12, 14]:
# 		run_each_dataset(parser.model_dir, retinanet, parser.dataset_path, 'test', 'MOT17-{:02d}'.format(seq_num))

if __name__ == '__main__':
	main()
