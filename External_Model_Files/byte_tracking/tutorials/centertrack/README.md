# CenterTrack

Step1.  git clone https://github.com/xingyizhou/CenterTrack.git


Step2. 

replace https://github.com/xingyizhou/CenterTrack/blob/master/src/lib/utils/tracker.py

replace https://github.com/xingyizhou/CenterTrack/blob/master/src/lib/opts.py


Step3. run
```
python3 test.py tracking --exp_id mot17_half --dataset mot --dataset_version 17halfval --pre_hm --ltrb_amodal  --load_model ../models/mot17_half.pth --track_thresh 0.4 --new_thresh 0.5 --out_thresh 0.2 --pre_thresh 0.5
```


# CenterTrack_BYTE

Step1.  git clone https://github.com/xingyizhou/CenterTrack.git


Step2. 

replace https://github.com/xingyizhou/CenterTrack/blob/master/src/lib/utils/tracker.py by byte_tracker.py

replace https://github.com/xingyizhou/CenterTrack/blob/master/src/lib/opts.py

add mot_online to https://github.com/xingyizhou/CenterTrack/blob/master/src/lib/utils

Step3. run
```
python3 test.py tracking --exp_id mot17_half --dataset mot --dataset_version 17halfval --pre_hm --ltrb_amodal  --load_model ../models/mot17_half.pth --track_thresh 0.4 --new_thresh 0.5 --out_thresh 0.2 --pre_thresh 0.5
```


## Notes
tracker.py: only motion

byte_tracker.py: motion with kalman filter

