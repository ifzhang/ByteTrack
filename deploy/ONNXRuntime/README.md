## ByteTrack-ONNXRuntime in Python

This doc introduces how to convert your pytorch model into onnx, and how to run an onnxruntime demo to verify your convertion.

### Convert Your Model to ONNX

```shell
cd <ByteTrack_HOME>
python3 tools/export_onnx.py --output-name bytetrack_s.onnx -f exps/example/mot/yolox_s_mix_det.py -c pretrained/bytetrack_s_mot17.pth.tar
```

### ONNXRuntime Demo

Step1.
```shell
cd <ByteTrack_HOME>/deploy/ONNXRuntime
```

Step2. 
```shell
python3 onnx_inference.py -m <ONNX_MODEL_PATH> -i <IMAGE_PATH> -o <OUTPUT_DIR> -s 0.3 --input_shape 640,640
```
