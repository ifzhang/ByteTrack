# MOTX
Every box matters for multi-object tracking

<summary>Installation</summary>

Step1. Install MOTX.
```shell
git clone https://github.com/ifzhang/MOTX.git
cd MOTX
pip3 install -r requirements.txt
python3 setup.py develop
```

Step2. Install [pycocotools](https://github.com/cocodataset/cocoapi).

```shell
pip3 install cython; pip3 install 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'
```

Step3. Others
```shell
pip3 install cython_bbox
```

<summary>Prepare datasets</summary>

Prepare coco format mot dataset.
```shell
cd <MOTX_HOME>
mkdir datasets
ln -s /path/to/your/mot ./datasets/mot
```
Change 'data_dir' in get_eval_loader() in exps/example/mot/yolox_x_ch.py ("mot" to "dancetrack")
```
data_dir=os.path.join(get_yolox_datadir(), "mot"),
```


<summary>Prepare pretrained models</summary>

```shell
cd <MOTX_HOME>
mkdir pretrained
cd pretrained
hdfs dfs -get hdfs://haruna/home/byte_arnold_hl_vc/zhangyifu/debug1/models/MOTX_pretrained.tar.gz
tar -zxvf MOTX_pretrained.tar.gz
```

<summary>Run tracking</summary>

Run MOTX:

```shell
cd <MOTX_HOME>
python3 tools/track.py -f exps/example/mot/yolox_x_ch_150.py -c pretrained/yolox_x_ch_150.pth.tar -b 1 -d 1 --fp16 --fuse
```

Run other trackers:
```shell
python3 tools/track_sort.py -f exps/example/mot/yolox_x_ch_150.py -c pretrained/yolox_x_ch_150.pth.tar -b 1 -d 1 --fp16 --fuse
python3 tools/track_deepsort.py -f exps/example/mot/yolox_x_ch_150.py -c pretrained/yolox_x_ch_150.pth.tar -b 1 -d 1 --fp16 --fuse
python3 tools/track_motdt.py -f exps/example/mot/yolox_x_ch_150.py -c pretrained/yolox_x_ch_150.pth.tar -b 1 -d 1 --fp16 --fuse
```

<summary>Train on CrowdHuman</summary>
Mix crowdhuman_train and crowdhuman_val and put the crowdhuman folder under <MOTX_HOME>/datasets.

```shell
cd <MOTX_HOME>
python3 tools/train.py -f exps/example/mot/yolox_x_ch_150.py -d 8 -b 48 --fp16 -o -c pretrained/yolox_x.pth
```

