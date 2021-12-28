# ByteTrack-TensorRT in C++

## Installation

Install opencv with ```sudo apt-get install libopencv-dev``` (we don't need a higher version of opencv like v3.3+).

Install eigen-3.3.9 [[google]](https://drive.google.com/file/d/1rqO74CYCNrmRAg8Rra0JP3yZtJ-rfket/view?usp=sharing), [[baidu(code:ueq4)]](https://pan.baidu.com/s/15kEfCxpy-T7tz60msxxExg).

```shell
unzip eigen-3.3.9.zip
cd eigen-3.3.9
mkdir build
cd build
cmake ..
sudo make install
```

## Prepare serialized engine file

Follow the TensorRT Python demo to convert and save the serialized engine file.

Check the 'model_trt.engine' file, which will be automatically saved at the YOLOX_output dir.

## Build the demo

You should set the TensorRT path and CUDA path in CMakeLists.txt.

For bytetrack_s model, we set the input frame size 1088 x 608. For bytetrack_m, bytetrack_l, bytetrack_x models, we set the input frame size 1440 x 800. You can modify the INPUT_W and INPUT_H in src/bytetrack.cpp

```c++
static const int INPUT_W = 1088;
static const int INPUT_H = 608;
```

You can first build the demo:

```shell
cd <ByteTrack_HOME>/deploy/TensorRT/cpp
mkdir build
cd build
cmake ..
make
```

Then you can run the demo with **200 FPS**:

```shell
./bytetrack ../../../../YOLOX_outputs/yolox_s_mix_det/model_trt.engine -i ../../../../videos/palace.mp4
```

(If you find the output video lose some frames, you can convert the input video by running:

```shell
cd <ByteTrack_HOME>
python3 tools/convert_video.py
```
to generate an appropriate input video for TensorRT C++ demo. )

