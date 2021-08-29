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


<summary>Prepare pretrained models</summary>
```shell
cd <MOTX_HOME>
mkdir pretrained
cd pretrained
hdfs dfs -get hdfs://haruna/home/byte_arnold_hl_vc/zhangyifu/debug1/models/MOTX_models.tar.gz
tar -zxvf MOTX_models.tar.gz
```

