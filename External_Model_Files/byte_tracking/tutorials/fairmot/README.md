# FairMOT

Step1.  git clone https://github.com/ifzhang/FairMOT.git


Step2. replace https://github.com/ifzhang/FairMOT/blob/master/src/lib/tracker/multitracker.py


Step3. run motion + reid tracker using tracker.py (set --match_thres 0.4), run BYTE tracker using byte_tracker.py (set --match_thres 0.8)

run BYTE tracker example: 
```
python3 track_half.py mot --load_model ../exp/mot/mot17_half_dla34/model_last.pth --match_thres 0.8
```


## Notes
byte_tracker: only motion

tracker: motion + reid
