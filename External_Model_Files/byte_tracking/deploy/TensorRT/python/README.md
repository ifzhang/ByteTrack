# ByteTrack-TensorRT in Python

## Install TensorRT Toolkit
Please follow the [TensorRT Installation Guide](https://docs.nvidia.com/deeplearning/tensorrt/install-guide/index.html) and [torch2trt gitrepo](https://github.com/NVIDIA-AI-IOT/torch2trt) to install TensorRT (Version 7 recommended) and torch2trt.

## Convert model

You can convert the Pytorch model “bytetrack_s_mot17” to TensorRT model by running:

```shell
cd <ByteTrack_HOME>
python3 tools/trt.py -f exps/example/mot/yolox_s_mix_det.py -c pretrained/bytetrack_s_mot17.pth.tar
```

## Run TensorRT demo

You can use the converted model_trt.pth to run TensorRT demo with **130 FPS**:

```shell
cd <ByteTrack_HOME>
python3 tools/demo_track.py video -f exps/example/mot/yolox_s_mix_det.py --trt --save_result
```
