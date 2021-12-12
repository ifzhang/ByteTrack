# CSTrack

Step1.  git clone -b MOT https://github.com/JudasDie/SOTS.git


Step2. replace https://github.com/JudasDie/SOTS/blob/master/lib/tracker/cstrack.py


Step3. download cstrack model trained on MIX and MOT17_half (mix_mot17_half_cstrack.pt): [google](https://drive.google.com/file/d/1OG5PDj_CYmMiw3dN6pZ0FsgqY__CIDx1/view?usp=sharing), [baidu(code:0bsu)](https://pan.baidu.com/s/1Z2VnE-OhZIPmgX6-4r9Z1Q) and put it under SOTS/weights


Step4. run BYTE tracker example:
```
cd tracking
python3 test_cstrack.py --val_mot17 True --val_hf 2 --weights ../weights/mix_mot17_half_cstrack.pt --conf_thres 0.6 --data_cfg ../src/lib/cfg/mot17_hf.json --data_dir your/data/path
```


## Notes
byte_tracker: only motion

tracker: motion + reid







