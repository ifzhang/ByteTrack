# TransTrack

Step1.  git clone https://github.com/PeizeSun/TransTrack.git


Step2. 

replace https://github.com/PeizeSun/TransTrack/blob/main/models/tracker.py

Step3.

Download TransTrack pretrained model: [671mot17_crowdhuman_mot17.pth](https://drive.google.com/drive/folders/1DjPL8xWoXDASrxgsA3O06EspJRdUXFQ-?usp=sharing)


Step3. run
```
python3 main_track.py  --output_dir . --dataset_file mot --coco_path mot --batch_size 1 --resume pretrained/671mot17_crowdhuman_mot17.pth --eval --with_box_refine --num_queries 500
```


# TransTrack_BYTE

Step1.  git clone https://github.com/PeizeSun/TransTrack.git

Step2. 

replace https://github.com/PeizeSun/TransTrack/blob/main/models/save_track.py

replace https://github.com/PeizeSun/TransTrack/blob/main/engine_track.py

replace https://github.com/PeizeSun/TransTrack/blob/main/main_track.py

add mot_online to https://github.com/PeizeSun/TransTrack

Step3. run
```
python3 main_track.py  --output_dir . --dataset_file mot --coco_path mot --batch_size 1 --resume pretrained/671mot17_crowdhuman_mot17.pth --eval --with_box_refine --num_queries 500
```


## Notes
tracker.py: only motion

mot_online/byte_tracker.py: motion with kalman filter

