# CTracker

Step1.  git clone https://github.com/pjl1995/CTracker.git and train on MOT17 train_half


Step2. 

replace https://github.com/pjl1995/CTracker/blob/master/test.py


Step3. run
```
python3 test.py --dataset_path MOT17_ROOT --model_dir ./trained_model/
```

Step4. 

add eval_motchallenge.py to https://github.com/pjl1995/CTracker

prepare gt_half_val.txt as CenterTrack [DATA.md](https://github.com/xingyizhou/CenterTrack/blob/master/readme/DATA.md)


Step5. run
```
python3 eval_motchallenge.py --groundtruths ../MOT17/train --tests results --gt_type half_val --eval_official  --score_threshold -1
```
