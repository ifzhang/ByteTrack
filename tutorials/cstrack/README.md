# CSTrack

Step1.  git clone https://github.com/JudasDie/SOTS.git


Step2. replace https://github.com/JudasDie/SOTS/blob/master/lib/tracker/cstrack.py


Step3. run motion tracker example:
```
python3 test_cstrack.py --val_mot17 True --val_hf 2 --weights your/weight/path --conf_thres 0.7 --data_cfg ../src/lib/cfg/mot17_hf.json --data_dir your/data/path
```


## Notes
motion_tracker: only motion

tracker: motion + reid







