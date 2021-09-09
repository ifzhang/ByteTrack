# CTracker

Step1.  git clone https://github.com/pjl1995/CTracker.git and train on MOT17 train_half


Step2. 

replace https://github.com/pjl1995/CTracker/blob/master/test.py


Step3. run
```
python3 test.py tracking --exp_id mot17_half --dataset mot --dataset_version 17halfval --pre_hm --ltrb_amodal  --load_model ../models/mot17_half.pth --track_thresh 0.4 --new_thresh 0.5 --out_thresh 0.2 --pre_thresh 0.5
```

Step4. 

add eval_motchallenge.py to https://github.com/pjl1995/CTracker

Step5. run
```
python3 eval_motchallenge.py --groundtruths ../MOT17/train --tests results --gt_type half-val --eval_official  --score_threshold -1
```
