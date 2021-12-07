# JDE

Step1.  git clone https://github.com/Zhongdao/Towards-Realtime-MOT.git


Step2. replace https://github.com/Zhongdao/Towards-Realtime-MOT/blob/master/tracker/multitracker.py and https://github.com/Zhongdao/Towards-Realtime-MOT/blob/master/utils/evaluation.py

Step3. download JDE model trained on MIX and MOT17_half (mix_mot17_half_jde.pt): [google](https://drive.google.com/file/d/1jUiIbaHFf75Jq6thOGI3CPygMMBy6850/view?usp=sharing), [baidu(code:ccdd)](https://pan.baidu.com/s/10se81ZktkUDUWn2dZzkk_Q)

Step4. put track_half.py under https://github.com/Zhongdao/Towards-Realtime-MOT and run:
```
python3 track_half.py --cfg ./cfg/yolov3_1088x608.cfg --weights weights/mix_mot17_half_jde.pt
```


## Notes
byte_tracker: only motion

tracker: motion + reid
