import os
import numpy as np
import json
from PIL import Image

DATA_PATH = 'datasets/crowdhuman/'
OUT_PATH = os.path.join(DATA_PATH, 'annotations')
SPLITS = ['val', 'train']
DEBUG = False

def load_func(fpath):
    print('Loading annotation file:', fpath)
    assert os.path.exists(fpath), f"Annotation file not found: {fpath}"
    with open(fpath, 'r') as fid:
        lines = fid.readlines()
    records = [json.loads(line.strip('\n')) for line in lines]
    return records

if __name__ == '__main__':
    if not os.path.exists(OUT_PATH):
        os.mkdir(OUT_PATH)
    for split in SPLITS:
        data_path = os.path.join(DATA_PATH, split)
        out_path = os.path.join(OUT_PATH, '{}.json'.format(split))
        out = {
            'images': [],
            'annotations': [],
            'categories': [{'id': 1, 'name': 'person'}]
        }
        ann_path = os.path.join(DATA_PATH, 'annotation_{}.odgt'.format(split))
        anns_data = load_func(ann_path)
        image_cnt = 0
        ann_cnt = 0

        for ann_data in anns_data:
            file_path = os.path.join(DATA_PATH, 'CrowdHuman_{}'.format(split), '{}.jpg'.format(ann_data['ID']))
            if not os.path.exists(file_path):
                print('Skipping missing file:', file_path)
                continue  # Skip this annotation if the file is not found
            try:
                im = Image.open(file_path)
            except Exception as e:
                print("Error opening file {}: {}".format(file_path, e))
                continue

            image_cnt += 1
            image_info = {
                'file_name': '{}.jpg'.format(ann_data['ID']),
                'id': image_cnt,
                'height': im.size[1],
                'width': im.size[0]
            }
            out['images'].append(image_info)
            
            if split != 'test':
                anns = ann_data.get('gtboxes', [])
                for ann in anns:
                    ann_cnt += 1
                    fbox = ann['fbox']
                    annotation = {
                        'id': ann_cnt,
                        'category_id': 1,
                        'image_id': image_cnt,
                        'track_id': -1,
                        'bbox_vis': ann['vbox'],
                        'bbox': fbox,
                        'area': fbox[2] * fbox[3],
                        'iscrowd': 1 if ('extra' in ann and 'ignore' in ann['extra'] and ann['extra']['ignore'] == 1) else 0
                    }
                    out['annotations'].append(annotation)
                    
        print('Loaded {}: {} images and {} annotations'.format(split, len(out['images']), len(out['annotations'])))
        with open(out_path, 'w') as fout:
            json.dump(out, fout)
