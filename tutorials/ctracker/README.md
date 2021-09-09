# CTracker

Step1.  git clone https://github.com/pjl1995/CTracker.git


Step2. 

add generate_half_csv.py to https://github.com/pjl1995/CTracker

run generate_half_csv.py

run
```
python3 train.py --root_path MOT17 --csv_train train_half_annots.csv --model_dir ./ctracker/ --depth 50
```

Step3. 

replace https://github.com/pjl1995/CTracker/blob/master/test.py

run
```
python3 test.py --dataset_path MOT17 --model_dir ./trained_model/
```

Step4. 

add eval_motchallenge.py to https://github.com/pjl1995/CTracker

prepare gt_half_val.txt as CenterTrack [DATA.md](https://github.com/xingyizhou/CenterTrack/blob/master/readme/DATA.md)


Step5. run
```
python3 eval_motchallenge.py --groundtruths MOT17/train --tests results --gt_type half_val --eval_official  --score_threshold -1
```
