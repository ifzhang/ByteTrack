# TraDeS

Step1.  git clone https://github.com/JialianW/TraDeS.git


Step2. 

replace https://github.com/JialianW/TraDeS/blob/master/src/lib/utils/tracker.py

replace https://github.com/JialianW/TraDeS/blob/master/src/lib/opts.py


Step3. run
```
python3 test.py tracking --exp_id mot17_half --dataset mot --dataset_version 17halfval --pre_hm --ltrb_amodal --inference --load_model ../models/mot_half.pth --gpus 0 --clip_len 3 --trades 
```


# TraDeS_kalman

Step1.  git clone https://github.com/JialianW/TraDeS.git


Step2. 

replace https://github.com/JialianW/TraDeS/blob/master/src/lib/utils/tracker.py by tracker_kalman.py

replace https://github.com/JialianW/TraDeS/blob/master/src/lib/opts.py

add mot_online to https://github.com/JialianW/TraDeS/blob/master/src/lib/utils

Step3. run
```
python3 test.py tracking --exp_id mot17_half --dataset mot --dataset_version 17halfval --pre_hm --ltrb_amodal --inference --load_model ../models/mot_half.pth --gpus 0 --clip_len 3 --trades 
```


## Notes
tracker.py: only motion

tracker_kalman.py: motion with kalman filter
