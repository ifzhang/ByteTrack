# QDTrack_reid_motion

Step1.  git clone https://github.com/SysCV/qdtrack.git and train


Step2. 

replace https://github.com/SysCV/qdtrack/blob/master/qdtrack/models/mot/qdtrack.py

add mot_online to https://github.com/SysCV/qdtrack

add tracker_reid_motion.py to https://github.com/SysCV/qdtrack and rename to tracker.py

Step3. download qdtrack model trained on mot17 half training set: [google](https://drive.google.com/file/d/1IfM8i0R0lF_4NOgeloMPFo5d52dqhaHW/view?usp=sharing), [baidu(code:whcc)](https://pan.baidu.com/s/1IYRD3V2YOa6-YNFgMQyv7w)

Step4. run
```
python3 -m torch.distributed.launch --nproc_per_node=8 --master_port=29501 tools/test.py configs/mot17/qdtrack-frcnn_r50_fpn_4e_mot17.py work_dirs/mot17_half_qdtrack.pth --launcher pytorch --eval track --eval-options resfile_path=output
```


# QDTrack_BYTE

Step1.  git clone https://github.com/SysCV/qdtrack.git and train


Step2. 

replace https://github.com/SysCV/qdtrack/blob/master/qdtrack/models/mot/qdtrack.py

add mot_online to https://github.com/SysCV/qdtrack

add byte_tracker.py to https://github.com/SysCV/qdtrack


Step3. run
```
python3 -m torch.distributed.launch --nproc_per_node=8 --master_port=29501 tools/test.py configs/mot17/qdtrack-frcnn_r50_fpn_4e_mot17.py work_dirs/mot17_half_qdtrack.pth --launcher pytorch --eval track --eval-options resfile_path=output
```
