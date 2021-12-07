## ByteTrack-ONNXRuntime in Python

This doc introduces how to convert your pytorch model into onnx, and how to run an onnxruntime demo to verify your convertion.

### Convert Your Model to ONNX

```shell
cd <ByteTrack_HOME>
python3 tools/export_onnx.py --output-name bytetrack_s.onnx -f exps/example/mot/yolox_s_mix_det.py -c pretrained/bytetrack_s_mot17.pth.tar
```

### ONNXRuntime Demo

You can run onnx demo with **16 FPS** (96-core Intel(R) Xeon(R) Platinum 8163 CPU @ 2.50GHz):

```shell
cd <ByteTrack_HOME>/deploy/ONNXRuntime
python3 onnx_inference.py
```
