import json
import os


"""
mkdir -p mix/annotations
cp mot/annotations/val_half.json mix/annotations/val_half.json
cp mot/annotations/test.json mix/annotations/test.json
cd mix
ln -s ../mot/train mot_train
ln -s ../crowdhuman/CrowdHuman_train crowdhuman_train
cd ..
"""

crowdhuman_json = json.load(open('datasets/crowdhuman/annotations/train.json','r'))
img_list = list()
ann_list = list()
video_list = list()
category_list = crowdhuman_json['categories']

img_id_count = 0
for img in crowdhuman_json['images']:
    img_id_count += 1
    img['file_name'] = 'crowdhuman_train/' + img['file_name']
    img['frame_id'] = img_id_count
    img['prev_image_id'] = img['id']
    img['next_image_id'] = img['id']
    img['video_id'] = 0
    img_list.append(img)
    
for ann in crowdhuman_json['annotations']:
    ann_list.append(ann)

video_list.append({
    'id': 0,
    'file_name': 'crowdhuman_train'
})


max_img = 30000
max_ann = 10000000

crowdhuman_val_json = json.load(open('datasets/crowdhuman/annotations/val.json','r'))
img_id_count = 0
for img in crowdhuman_val_json['images']:
    img_id_count += 1
    img['file_name'] = 'crowdhuman_val/' + img['file_name']
    img['frame_id'] = img_id_count
    img['prev_image_id'] = img['id'] + max_img
    img['next_image_id'] = img['id'] + max_img
    img['id'] = img['id'] + max_img
    img['video_id'] = 0
    img_list.append(img)
    
for ann in crowdhuman_val_json['annotations']:
    ann['id'] = ann['id'] + max_ann
    ann['image_id'] = ann['image_id'] + max_img
    ann_list.append(ann)

video_list.append({
    'id': 0,
    'file_name': 'crowdhuman_val'
})

mix_json = dict()
mix_json['images'] = img_list
mix_json['annotations'] = ann_list
mix_json['videos'] = video_list
mix_json['categories'] = category_list
json.dump(mix_json, open('datasets/ch_all/annotations/train.json','w'))