# ByteTrack

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/bytetrack-multi-object-tracking-by-1/multi-object-tracking-on-mot17)](https://paperswithcode.com/sota/multi-object-tracking-on-mot17?p=bytetrack-multi-object-tracking-by-1)

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/bytetrack-multi-object-tracking-by-1/multi-object-tracking-on-mot20-1)](https://paperswithcode.com/sota/multi-object-tracking-on-mot20-1?p=bytetrack-multi-object-tracking-by-1)

#### ByteTrack is a simple, fast and strong multi-object tracker.

<p align="center"><img src="assets/sota.png" width="500"/></p>

> [**ByteTrack: Multi-Object Tracking by Associating Every Detection Box**](https://arxiv.org/abs/2110.06864)
> 
> Yifu Zhang, Peize Sun, Yi Jiang, Dongdong Yu, Zehuan Yuan, Ping Luo, Wenyu Liu, Xinggang Wang
> 
> *[arXiv 2110.06864](https://arxiv.org/abs/2110.06864)*

## Demo Links
| Google Colab demo | Huggingface Demo | Original Paper: ByteTrack |
|:-:|:-:|:-:|
|[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1bDilg4cmXFa8HCKHbsZ_p16p0vrhLyu0?usp=sharing)|[![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/akhaliq/bytetrack)|[arXiv 2110.06864](https://arxiv.org/abs/2110.06864)|
* Integrated to [Huggingface Spaces](https://huggingface.co/spaces) with [Gradio](https://github.com/gradio-app/gradio).


## Abstract
Multi-object tracking (MOT) aims at estimating bounding boxes and identities of objects in videos. Most methods obtain identities by associating detection boxes whose scores are higher than a threshold. The objects with low detection scores, e.g. occluded objects, are simply thrown away, which brings non-negligible true object missing and fragmented trajectories. To solve this problem, we present a simple, effective and generic association method, tracking by associating every detection box instead of only the high score ones. For the low score detection boxes, we utilize their similarities with tracklets to recover true objects and filter out the background detections. When applied to 9 different state-of-the-art trackers, our method achieves consistent improvement on IDF1 scores ranging from 1 to 10 points. To put forwards the state-of-the-art performance of MOT, we design a simple and strong tracker, named ByteTrack. For the first time, we achieve 80.3 MOTA, 77.3 IDF1 and 63.1 HOTA on the test set of MOT17 with 30 FPS running speed on a single V100 GPU.
<p align="center"><img src="assets/teasing.png" width="400"/></p>

## Tracking performance
### Results on MOT challenge test set
| Dataset    |  MOTA | IDF1 | HOTA | MT | ML | FP | FN | IDs | FPS |
|------------|-------|------|------|-------|-------|------|------|------|------|
|MOT17       | 80.3 | 77.3 | 63.1 | 53.2% | 14.5% | 25491 | 83721 | 2196 | 29.6 |
|MOT20       | 77.8 | 75.2 | 61.3 | 69.2% | 9.5%  | 26249 | 87594 | 1223 | 13.7 |

### Visualization results on MOT challenge test set
<img src="assets/MOT17-01-SDP.gif" width="400"/>   <img src="assets/MOT17-07-SDP.gif" width="400"/>
<img src="assets/MOT20-07.gif" width="400"/>   <img src="assets/MOT20-08.gif" width="400"/>

## Installation
### 1. Installing on the host machine
Step1. Install ByteTrack.
```shell
git clone https://github.com/ifzhang/ByteTrack.git
cd ByteTrack
pip3 install -r requirements.txt
python3 setup.py develop
```

Step2. Install [pycocotools](https://github.com/cocodataset/cocoapi).

```shell
pip3 install cython; pip3 install 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'
```

Step3. Others
```shell
pip3 install cython_bbox
```
### 2. Docker build
```shell
docker build -t bytetrack:latest .

# Startup sample
mkdir -p pretrained && \
mkdir -p YOLOX_outputs && \
xhost +local: && \
docker run --gpus all -it --rm \
-v $PWD/pretrained:/workspace/ByteTrack/pretrained \
-v $PWD/datasets:/workspace/ByteTrack/datasets \
-v $PWD/YOLOX_outputs:/workspace/ByteTrack/YOLOX_outputs \
-v /tmp/.X11-unix/:/tmp/.X11-unix:rw \
--device /dev/video0:/dev/video0:mwr \
--net=host \
-e XDG_RUNTIME_DIR=$XDG_RUNTIME_DIR \
-e DISPLAY=$DISPLAY \
--privileged \
bytetrack:latest
```

## Data preparation

Download [MOT17](https://motchallenge.net/), [MOT20](https://motchallenge.net/), [CrowdHuman](https://www.crowdhuman.org/), [Cityperson](https://github.com/Zhongdao/Towards-Realtime-MOT/blob/master/DATASET_ZOO.md), [ETHZ](https://github.com/Zhongdao/Towards-Realtime-MOT/blob/master/DATASET_ZOO.md) and put them under <ByteTrack_HOME>/datasets in the following structure:
```
datasets
   |——————mot
   |        └——————train
   |        └——————test
   └——————crowdhuman
   |         └——————Crowdhuman_train
   |         └——————Crowdhuman_val
   |         └——————annotation_train.odgt
   |         └——————annotation_val.odgt
   └——————MOT20
   |        └——————train
   |        └——————test
   └——————Cityscapes
   |        └——————images
   |        └——————labels_with_ids
   └——————ETHZ
            └——————eth01
            └——————...
            └——————eth07
```

Then, you need to turn the datasets to COCO format and mix different training data:

```shell
cd <ByteTrack_HOME>
python3 tools/convert_mot17_to_coco.py
python3 tools/convert_mot20_to_coco.py
python3 tools/convert_crowdhuman_to_coco.py
python3 tools/convert_cityperson_to_coco.py
python3 tools/convert_ethz_to_coco.py
```

Before mixing different datasets, you need to follow the operations in [mix_xxx.py](https://github.com/ifzhang/ByteTrack/blob/c116dfc746f9ebe07d419caa8acba9b3acfa79a6/tools/mix_data_ablation.py#L6) to create a data folder and link. Finally, you can mix the training data:

```shell
cd <ByteTrack_HOME>
python3 tools/mix_data_ablation.py
python3 tools/mix_data_test_mot17.py
python3 tools/mix_data_test_mot20.py
```


## Model zoo

### Ablation model

Train on CrowdHuman and MOT17 half train, evaluate on MOT17 half val

| Model    |  MOTA | IDF1 | IDs | FPS |
|------------|-------|------|------|------|
|ByteTrack_ablation [[google]](https://drive.google.com/file/d/1iqhM-6V_r1FpOlOzrdP_Ejshgk0DxOob/view?usp=sharing), [[baidu(code:eeo8)]](https://pan.baidu.com/s/1W5eRBnxc4x9V8gm7dgdEYg) | 76.6 | 79.3 | 159 | 29.6 |

### MOT17 test model

Train on CrowdHuman, MOT17, Cityperson and ETHZ, evaluate on MOT17 train.

* **Standard models**

| Model    |  MOTA | IDF1 | IDs | FPS |
|------------|-------|------|------|------|
|bytetrack_x_mot17 [[google]](https://drive.google.com/file/d/1P4mY0Yyd3PPTybgZkjMYhFri88nTmJX5/view?usp=sharing), [[baidu(code:ic0i)]](https://pan.baidu.com/s/1OJKrcQa_JP9zofC6ZtGBpw) | 90.0 | 83.3 | 422 | 29.6 |
|bytetrack_l_mot17 [[google]](https://drive.google.com/file/d/1XwfUuCBF4IgWBWK2H7oOhQgEj9Mrb3rz/view?usp=sharing), [[baidu(code:1cml)]](https://pan.baidu.com/s/1242adimKM6TYdeLU2qnuRA) | 88.7 | 80.7 | 460 | 43.7 |
|bytetrack_m_mot17 [[google]](https://drive.google.com/file/d/11Zb0NN_Uu7JwUd9e6Nk8o2_EUfxWqsun/view?usp=sharing), [[baidu(code:u3m4)]](https://pan.baidu.com/s/1fKemO1uZfvNSLzJfURO4TQ) | 87.0 | 80.1 | 477 | 54.1 |
|bytetrack_s_mot17 [[google]](https://drive.google.com/file/d/1uSmhXzyV1Zvb4TJJCzpsZOIcw7CCJLxj/view?usp=sharing), [[baidu(code:qflm)]](https://pan.baidu.com/s/1PiP1kQfgxAIrnGUbFP6Wfg) | 79.2 | 74.3 | 533 | 64.5 |

* **Light models**

| Model    |  MOTA | IDF1 | IDs | Params(M) | FLOPs(G) |
|------------|-------|------|------|------|-------|
|bytetrack_nano_mot17 [[google]](https://drive.google.com/file/d/1AoN2AxzVwOLM0gJ15bcwqZUpFjlDV1dX/view?usp=sharing), [[baidu(code:1ub8)]](https://pan.baidu.com/s/1dMxqBPP7lFNRZ3kFgDmWdw) | 69.0 | 66.3 | 531 | 0.90 | 3.99 |
|bytetrack_tiny_mot17 [[google]](https://drive.google.com/file/d/1LFAl14sql2Q5Y9aNFsX_OqsnIzUD_1ju/view?usp=sharing), [[baidu(code:cr8i)]](https://pan.baidu.com/s/1jgIqisPSDw98HJh8hqhM5w) | 77.1 | 71.5 | 519 | 5.03 | 24.45 |



### MOT20 test model

Train on CrowdHuman and MOT20, evaluate on MOT20 train.


| Model    |  MOTA | IDF1 | IDs | FPS |
|------------|-------|------|------|------|
|bytetrack_x_mot20 [[google]](https://drive.google.com/file/d/1HX2_JpMOjOIj1Z9rJjoet9XNy_cCAs5U/view?usp=sharing), [[baidu(code:3apd)]](https://pan.baidu.com/s/1bowJJj0bAnbhEQ3_6_Am0A) | 93.4 | 89.3 | 1057 | 17.5 |


## Training

The COCO pretrained YOLOX model can be downloaded from their [model zoo](https://github.com/Megvii-BaseDetection/YOLOX/tree/0.1.0). After downloading the pretrained models, you can put them under <ByteTrack_HOME>/pretrained.

* **Train ablation model (MOT17 half train and CrowdHuman)**

```shell
cd <ByteTrack_HOME>
python3 tools/train.py -f exps/example/mot/yolox_x_ablation.py -d 8 -b 48 --fp16 -o -c pretrained/yolox_x.pth
```

* **Train MOT17 test model (MOT17 train, CrowdHuman, Cityperson and ETHZ)**

```shell
cd <ByteTrack_HOME>
python3 tools/train.py -f exps/example/mot/yolox_x_mix_det.py -d 8 -b 48 --fp16 -o -c pretrained/yolox_x.pth
```

* **Train MOT20 test model (MOT20 train, CrowdHuman)**

For MOT20, you need to clip the bounding boxes inside the image.

Add clip operation in [line 134-135 in data_augment.py](https://github.com/ifzhang/ByteTrack/blob/72cd6dd24083c337a9177e484b12bb2b5b3069a6/yolox/data/data_augment.py#L134), [line 122-125 in mosaicdetection.py](https://github.com/ifzhang/ByteTrack/blob/72cd6dd24083c337a9177e484b12bb2b5b3069a6/yolox/data/datasets/mosaicdetection.py#L122), [line 217-225 in mosaicdetection.py](https://github.com/ifzhang/ByteTrack/blob/72cd6dd24083c337a9177e484b12bb2b5b3069a6/yolox/data/datasets/mosaicdetection.py#L217), [line 115-118 in boxes.py](https://github.com/ifzhang/ByteTrack/blob/72cd6dd24083c337a9177e484b12bb2b5b3069a6/yolox/utils/boxes.py#L115).

```shell
cd <ByteTrack_HOME>
python3 tools/train.py -f exps/example/mot/yolox_x_mix_mot20_ch.py -d 8 -b 48 --fp16 -o -c pretrained/yolox_x.pth
```

* **Train custom dataset**

First, you need to prepare your dataset in COCO format. You can refer to [MOT-to-COCO](https://github.com/ifzhang/ByteTrack/blob/main/tools/convert_mot17_to_coco.py) or [CrowdHuman-to-COCO](https://github.com/ifzhang/ByteTrack/blob/main/tools/convert_crowdhuman_to_coco.py). Then, you need to create a Exp file for your dataset. You can refer to the [CrowdHuman](https://github.com/ifzhang/ByteTrack/blob/main/exps/example/mot/yolox_x_ch.py) training Exp file. Don't forget to modify get_data_loader() and get_eval_loader in your Exp file. Finally, you can train bytetrack on your dataset by running:

```shell
cd <ByteTrack_HOME>
python3 tools/train.py -f exps/example/mot/your_exp_file.py -d 8 -b 48 --fp16 -o -c pretrained/yolox_x.pth
```


## Tracking

* **Evaluation on MOT17 half val**

Run ByteTrack:

```shell
cd <ByteTrack_HOME>
python3 tools/track.py -f exps/example/mot/yolox_x_ablation.py -c pretrained/bytetrack_ablation.pth.tar -b 1 -d 1 --fp16 --fuse
```
You can get 76.6 MOTA using our pretrained model.

Run other trackers:
```shell
python3 tools/track_sort.py -f exps/example/mot/yolox_x_ablation.py -c pretrained/bytetrack_ablation.pth.tar -b 1 -d 1 --fp16 --fuse
python3 tools/track_deepsort.py -f exps/example/mot/yolox_x_ablation.py -c pretrained/bytetrack_ablation.pth.tar -b 1 -d 1 --fp16 --fuse
python3 tools/track_motdt.py -f exps/example/mot/yolox_x_ablation.py -c pretrained/bytetrack_ablation.pth.tar -b 1 -d 1 --fp16 --fuse
```

* **Test on MOT17**

Run ByteTrack:

```shell
cd <ByteTrack_HOME>
python3 tools/track.py -f exps/example/mot/yolox_x_mix_det.py -c pretrained/bytetrack_x_mot17.pth.tar -b 1 -d 1 --fp16 --fuse
python3 tools/interpolation.py
```
Submit the txt files to [MOTChallenge](https://motchallenge.net/) website and you can get 79+ MOTA (For 80+ MOTA, you need to carefully tune the test image size and high score detection threshold of each sequence).

* **Test on MOT20**

We use the input size 1600 x 896 for MOT20-04, MOT20-07 and 1920 x 736 for MOT20-06, MOT20-08. You can edit it in [yolox_x_mix_mot20_ch.py](https://github.com/ifzhang/ByteTrack/blob/main/exps/example/mot/yolox_x_mix_mot20_ch.py)

Run ByteTrack:

```shell
cd <ByteTrack_HOME>
python3 tools/track.py -f exps/example/mot/yolox_x_mix_mot20_ch.py -c pretrained/bytetrack_x_mot20.pth.tar -b 1 -d 1 --fp16 --fuse --match_thresh 0.7 --mot20
python3 tools/interpolation.py
```
Submit the txt files to [MOTChallenge](https://motchallenge.net/) website and you can get 77+ MOTA (For higher MOTA, you need to carefully tune the test image size and high score detection threshold of each sequence).

## Applying BYTE to other trackers

See [tutorials](https://github.com/ifzhang/ByteTrack/tree/main/tutorials).

## Combining BYTE with other detectors

Suppose you have already got the detection results 'dets' (x1, y1, x2, y2, score) from other detectors, you can simply pass the detection results to BYTETracker (you need to first modify some post-processing code according to the format of your detection results in [byte_tracker.py](https://github.com/ifzhang/ByteTrack/blob/main/yolox/tracker/byte_tracker.py)):

```
from yolox.tracker.byte_tracker import BYTETracker
tracker = BYTETracker(args)
for image in images:
   dets = detector(image)
   online_targets = tracker.update(dets, info_imgs, img_size)
```

You can get the tracking results in each frame from 'online_targets'. You can refer to [mot_evaluators.py](https://github.com/ifzhang/ByteTrack/blob/main/yolox/evaluators/mot_evaluator.py) to pass the detection results to BYTETracker.

## Demo

<img src="assets/palace_demo.gif" width="600"/>

```shell
cd <ByteTrack_HOME>
python3 tools/demo_track.py video -f exps/example/mot/yolox_x_mix_det.py -c pretrained/bytetrack_x_mot17.pth.tar --fp16 --fuse --save_result
```

## Deploy

1.  [ONNX export and ONNXRuntime](./deploy/ONNXRuntime)
2.  [TensorRT in Python](./deploy/TensorRT/python)
3.  [TensorRT in C++](./deploy/TensorRT/cpp)
4.  [ncnn in C++](./deploy/ncnn/cpp)

## Citation

```
@article{zhang2021bytetrack,
  title={ByteTrack: Multi-Object Tracking by Associating Every Detection Box},
  author={Zhang, Yifu and Sun, Peize and Jiang, Yi and Yu, Dongdong and Yuan, Zehuan and Luo, Ping and Liu, Wenyu and Wang, Xinggang},
  journal={arXiv preprint arXiv:2110.06864},
  year={2021}
}
```

## Acknowledgement

A large part of the code is borrowed from [YOLOX](https://github.com/Megvii-BaseDetection/YOLOX), [FairMOT](https://github.com/ifzhang/FairMOT), [TransTrack](https://github.com/PeizeSun/TransTrack) and [JDE-Cpp](https://github.com/samylee/Towards-Realtime-MOT-Cpp). Many thanks for their wonderful works.
