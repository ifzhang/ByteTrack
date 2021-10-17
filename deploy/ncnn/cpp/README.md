# ByteTrack-CPP-ncnn

## Installation

Clone [ncnn](https://github.com/Tencent/ncnn) first, then please following [build tutorial of ncnn](https://github.com/Tencent/ncnn/wiki/how-to-build) to build on your own device.

Install eigen-3.3.9 [[google]](https://drive.google.com/file/d/1rqO74CYCNrmRAg8Rra0JP3yZtJ-rfket/view?usp=sharing), [[baidu(code:ueq4)]](https://pan.baidu.com/s/15kEfCxpy-T7tz60msxxExg).

```shell
unzip eigen-3.3.9.zip
cd eigen-3.3.9
mkdir build
cd build
cmake ..
sudo make install
```

## Generate onnx file
Use provided tools to generate onnx file.
For example, if you want to generate onnx file of bytetrack_s_mot17.pth, please run the following command:
```shell
cd <ByteTrack_HOME>
python3 tools/export_onnx.py -f exps/example/mot/yolox_s_mix_det.py -c pretrained/bytetrack_s_mot17.pth.tar
```
Then, a bytetrack_s.onnx file is generated under <ByteTrack_HOME>.

## Generate ncnn param and bin file
Put bytetrack_s.onnx under ncnn/build/tools/onnx and then run: 

```shell
cd ncnn/build/tools/onnx
./onnx2ncnn bytetrack_s.onnx bytetrack_s.param bytetrack_s.bin
```

Since Focus module is not supported in ncnn. Warnings like:
```shell
Unsupported slice step ! 
```
will be printed. However, don't  worry!  C++ version of Focus layer is already implemented in src/bytetrack.cpp.
  
## Modify param file
Open **bytetrack_s.param**, and modify it.
Before (just an example):
```
235 268
Input            images                   0 1 images
Split            splitncnn_input0         1 4 images images_splitncnn_0 images_splitncnn_1 images_splitncnn_2 images_splitncnn_3
Crop             Slice_4                  1 1 images_splitncnn_3 467 -23309=1,0 -23310=1,2147483647 -23311=1,1
Crop             Slice_9                  1 1 467 472 -23309=1,0 -23310=1,2147483647 -23311=1,2
Crop             Slice_14                 1 1 images_splitncnn_2 477 -23309=1,0 -23310=1,2147483647 -23311=1,1
Crop             Slice_19                 1 1 477 482 -23309=1,1 -23310=1,2147483647 -23311=1,2
Crop             Slice_24                 1 1 images_splitncnn_1 487 -23309=1,1 -23310=1,2147483647 -23311=1,1
Crop             Slice_29                 1 1 487 492 -23309=1,0 -23310=1,2147483647 -23311=1,2
Crop             Slice_34                 1 1 images_splitncnn_0 497 -23309=1,1 -23310=1,2147483647 -23311=1,1
Crop             Slice_39                 1 1 497 502 -23309=1,1 -23310=1,2147483647 -23311=1,2
Concat           Concat_40                4 1 472 492 482 502 503 0=0
...
```
* Change first number for 235 to 235 - 9 = 226(since we will remove 10 layers and add 1 layers, total layers number should minus 9). 
* Then remove 10 lines of code from Split to Concat, but remember the last but 2nd number: 503.
* Add YoloV5Focus layer After Input (using previous number 503):
```
YoloV5Focus      focus                    1 1 images 503
```
After(just an exmaple):
```
226 328
Input            images                   0 1 images
YoloV5Focus      focus                    1 1 images 503
...
```

## Use ncnn_optimize to generate new param and bin
```shell
# suppose you are still under ncnn/build/tools/onnx dir.
../ncnnoptimize bytetrack_s.param bytetrack_s.bin bytetrack_s_op.param bytetrack_s_op.bin 65536
```

## Copy files and build ByteTrack
Copy or move 'src', 'include' folders and 'CMakeLists.txt' file into ncnn/examples. Copy bytetrack_s_op.param, bytetrack_s_op.bin and <ByteTrack_HOME>/videos/palace.mp4 into ncnn/build/examples. Then, build ByteTrack:

```shell
cd ncnn/build/examples
cmake ..
make
```

## Run the demo
You can run the ncnn demo with **5 FPS** (96-core Intel(R) Xeon(R) Platinum 8163 CPU @ 2.50GHz):
```shell
./bytetrack palace.mp4
```

You can modify 'num_threads' to optimize the running speed in [bytetrack.cpp](https://github.com/ifzhang/ByteTrack/blob/2e9a67895da6b47b948015f6861bba0bacd4e72f/deploy/ncnn/cpp/src/bytetrack.cpp#L309) according to the number of your CPU cores:

```
yolox.opt.num_threads = 20;
```


## Acknowledgement

* [ncnn](https://github.com/Tencent/ncnn)
