# CSTrack

Step1.  git clone https://github.com/JudasDie/SOTS.git


Step2. replace https://github.com/JudasDie/SOTS/blob/master/lib/tracker/cstrack.py


Step3. download cstrack model trained on MIX and MOT17_half (mix_mot17_half_cstrack.pt): [google](https://drive.google.com/file/d/1OG5PDj_CYmMiw3dN6pZ0FsgqY__CIDx1/view?usp=sharing), [baidu,code:0bsu](https://pan.baidu.com/s/1Z2VnE-OhZIPmgX6-4r9Z1Q)


Step4. run motion tracker example:
```
python3 test_cstrack.py --val_mot17 True --val_hf 2 --weights weights/mix_mot17_half_cstrack.pt --conf_thres 0.7 --data_cfg ../src/lib/cfg/mot17_hf.json --data_dir your/data/path
```


## Notes
motion_tracker: only motion

tracker: motion + reid







