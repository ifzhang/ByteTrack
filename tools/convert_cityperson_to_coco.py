import os
import numpy as np
import json
from PIL import Image

DATA_PATH = 'datasets/Cityscapes/'
DATA_FILE_PATH = 'datasets/data_path/citypersons.train'
OUT_PATH = DATA_PATH + 'annotations/'

def load_paths(data_path):
    with open(data_path, 'r') as file:
        img_files = file.readlines()
        img_files = [x.replace('\n', '') for x in img_files]
        img_files = list(filter(lambda x: len(x) > 0, img_files))
    label_files = [x.replace('images', 'labels_with_ids').replace('.png', '.txt').replace('.jpg', '.txt') for x in img_files]
    return img_files, label_files                    

if __name__ == '__main__':
    if not os.path.exists(OUT_PATH):
        os.mkdir(OUT_PATH)

    out_path = OUT_PATH + 'train.json'
    out = {'images': [], 'annotations': [], 'categories': [{'id': 1, 'name': 'person'}]}
    img_paths, label_paths = load_paths(DATA_FILE_PATH)
    image_cnt = 0
    ann_cnt = 0
    video_cnt = 0
    for img_path, label_path in zip(img_paths, label_paths):
        image_cnt += 1
        im = Image.open(os.path.join("datasets", img_path))
        image_info = {'file_name': img_path, 
                        'id': image_cnt,
                        'height': im.size[1], 
                        'width': im.size[0]}
        out['images'].append(image_info)
        # Load labels
        if os.path.isfile(os.path.join("datasets", label_path)):
            labels0 = np.loadtxt(os.path.join("datasets", label_path), dtype=np.float32).reshape(-1, 6)
            # Normalized xywh to pixel xyxy format
            labels = labels0.copy()
            labels[:, 2] = image_info['width'] * (labels0[:, 2] - labels0[:, 4] / 2)
            labels[:, 3] = image_info['height'] * (labels0[:, 3] - labels0[:, 5] / 2)
            labels[:, 4] = image_info['width'] * labels0[:, 4]
            labels[:, 5] = image_info['height'] * labels0[:, 5]
        else:
            labels = np.array([])
        for i in range(len(labels)):
            ann_cnt += 1
            fbox = labels[i, 2:6].tolist()
            ann = {'id': ann_cnt,
                    'category_id': 1,
                    'image_id': image_cnt,
                    'track_id': -1,
                    'bbox': fbox,
                    'area': fbox[2] * fbox[3],
                    'iscrowd': 0}
            out['annotations'].append(ann)
    print('loaded train for {} images and {} samples'.format(len(out['images']), len(out['annotations'])))
    json.dump(out, open(out_path, 'w'))
