# QDTrack_reid_motion

Step1.  git clone https://github.com/SysCV/qdtrack.git and train


Step2. 

replace https://github.com/SysCV/qdtrack/blob/master/qdtrack/models/mot/qdtrack.py

add mot_online to https://github.com/SysCV/qdtrack

add tracker_reid_motion.py to https://github.com/SysCV/qdtrack and rename to tracker.py

Step3. run
```
python3 -m torch.distributed.launch --nproc_per_node=8 --master_port=29501 tools/test.py configs/mot17/qdtrack-frcnn_r50_fpn_4e_mot17.py work_dirs/MOT17/qdtrack-frcnn_r50_fpn_4e_mot17/latest.pth --launcher pytorch --eval track --eval-options resfile_path=output
```


# QDTrack_BYTE

Step1.  git clone https://github.com/SysCV/qdtrack.git and train


Step2. 

replace https://github.com/SysCV/qdtrack/blob/master/qdtrack/models/mot/qdtrack.py

add mot_online to https://github.com/SysCV/qdtrack

add tracker.py to https://github.com/SysCV/qdtrack


Step3. run
```
python3 -m torch.distributed.launch --nproc_per_node=8 --master_port=29501 tools/test.py configs/mot17/qdtrack-frcnn_r50_fpn_4e_mot17.py work_dirs/MOT17/qdtrack-frcnn_r50_fpn_4e_mot17/latest.pth --launcher pytorch --eval track --eval-options resfile_path=output
```
