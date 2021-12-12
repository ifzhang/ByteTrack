import os
import numpy as np
import json
from PIL import Image

DATA_PATH = 'datasets/crowdhuman/'
OUT_PATH = DATA_PATH + 'annotations/'
SPLITS = ['val', 'train']
DEBUG = False

def load_func(fpath):
    print('fpath', fpath)
    assert os.path.exists(fpath)
    with open(fpath,'r') as fid:
        lines = fid.readlines()
    records =[json.loads(line.strip('\n')) for line in lines]
    return records

if __name__ == '__main__':
    if not os.path.exists(OUT_PATH):
        os.mkdir(OUT_PATH)
    for split in SPLITS:
        data_path = DATA_PATH + split
        out_path = OUT_PATH + '{}.json'.format(split)
        out = {'images': [], 'annotations': [], 'categories': [{'id': 1, 'name': 'person'}]}
        ann_path = DATA_PATH + 'annotation_{}.odgt'.format(split)
        anns_data = load_func(ann_path)
        image_cnt = 0
        ann_cnt = 0
        video_cnt = 0
        for ann_data in anns_data:
            image_cnt += 1
            file_path = DATA_PATH + 'CrowdHuman_{}/'.format(split) + '{}.jpg'.format(ann_data['ID'])
            im = Image.open(file_path)
            image_info = {'file_name': '{}.jpg'.format(ann_data['ID']), 
                          'id': image_cnt,
                          'height': im.size[1], 
                          'width': im.size[0]}
            out['images'].append(image_info)
            if split != 'test':
                anns = ann_data['gtboxes']
                for i in range(len(anns)):
                    ann_cnt += 1
                    fbox = anns[i]['fbox']
                    ann = {'id': ann_cnt,
                         'category_id': 1,
                         'image_id': image_cnt,
                         'track_id': -1,
                         'bbox_vis': anns[i]['vbox'],
                         'bbox': fbox,
                         'area': fbox[2] * fbox[3],
                         'iscrowd': 1 if 'extra' in anns[i] and \
                                         'ignore' in anns[i]['extra'] and \
                                         anns[i]['extra']['ignore'] == 1 else 0}
                    out['annotations'].append(ann)
        print('loaded {} for {} images and {} samples'.format(split, len(out['images']), len(out['annotations'])))
        json.dump(out, open(out_path, 'w'))