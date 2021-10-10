# MOTR

Step1.  

git clone https://github.com/megvii-model/MOTR.git and install

replace https://github.com/megvii-model/MOTR/blob/main/datasets/joint.py

replace https://github.com/megvii-model/MOTR/blob/main/datasets/transforms.py


train

```
python3 -m torch.distributed.launch --nproc_per_node=8 \
    --use_env main.py \
    --meta_arch motr \
    --dataset_file e2e_joint \
    --epoch 50 \
    --with_box_refine \
    --lr_drop 40 \
    --lr 2e-4 \
    --lr_backbone 2e-5 \
    --pretrained coco_model_final.pth \
    --output_dir exps/e2e_motr_r50_mot17trainhalf \
    --batch_size 1 \
    --sample_mode 'random_interval' \
    --sample_interval 10 \
    --sampler_steps 10 20 30 \
    --sampler_lengths 2 3 4 5 \
    --update_query_pos \
    --merger_dropout 0 \
    --dropout 0 \
    --random_drop 0.1 \
    --fp_ratio 0.3 \
    --query_interaction_layer 'QIM' \
    --extra_track_attn \
    --mot_path .
    --data_txt_path_train ./datasets/data_path/mot17.half \
    --data_txt_path_val ./datasets/data_path/mot17.val \
```
mot17.half and mot17.val are from https://github.com/ifzhang/FairMOT/tree/master/src/data

You can also download the MOTR model trained by us: [google](https://drive.google.com/file/d/1pzGi53VooppQqhKf3TSxLK99LERsVyTw/view?usp=sharing), [baidu(code:t87h)](https://pan.baidu.com/s/1OrcR3L9Bf2xXIo8RQl3zyA)


Step2. 
   
replace https://github.com/megvii-model/MOTR/blob/main/util/evaluation.py

replace https://github.com/megvii-model/MOTR/blob/main/eval.py

replace https://github.com/megvii-model/MOTR/blob/main/models/motr.py

add byte_tracker.py to https://github.com/megvii-model/MOTR

add mot_online to https://github.com/megvii-model/MOTR


Step3. 


val

```
python3 eval.py \
    --meta_arch motr \
    --dataset_file e2e_joint \
    --epoch 200 \
    --with_box_refine \
    --lr_drop 100 \
    --lr 2e-4 \
    --lr_backbone 2e-5 \
    --pretrained exps/e2e_motr_r50_mot17val/motr_final.pth \
    --output_dir exps/e2e_motr_r50_mot17val \
    --batch_size 1 \
    --sample_mode 'random_interval' \
    --sample_interval 10 \
    --sampler_steps 50 90 120 \
    --sampler_lengths 2 3 4 5 \
    --update_query_pos \
    --merger_dropout 0 \
    --dropout 0 \
    --random_drop 0.1 \
    --fp_ratio 0.3 \
    --query_interaction_layer 'QIM' \
    --extra_track_attn \
    --mot_path ./MOT17/images/train
    --data_txt_path_train ./datasets/data_path/mot17.half \
    --data_txt_path_val ./datasets/data_path/mot17.val \
    --resume model_final.pth \
```



# MOTR det

in Step2, replace https://github.com/megvii-model/MOTR/blob/main/models/motr.py by motr_det.py 

others are the same as MOTR
