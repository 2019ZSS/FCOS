## Pytorch implement FCOS Model

###  Main Work

1. *Pytorch implement FCOS Model for Pascal VOC 2006 and coco 2017*
2. *Design a SIMO network for FCOS*
3. *Add DCNv2 for FCOS*
4. *Add ASFF for FCOS*
5. *Add IRCNN for FCOS*

### Install  

```bash
git clone https://github.com/2019ZSS/FCOS.git
```

### Environment

```bash
conda create -n FCOS python=3.7 pytorch torchvision cudatoolkit=10.1 -c pytorch
conda activate FCOS
pip install opencv-python
pip install pycocotools
pip install cython
pip install Pillow
pip install efficientnet_pytorch
pip install cupy

(optional if you dont'need to run Dcnv2)
cd FCOS/model/DCNv2
python setup.py install
```

### Data

pwd follow:

```bash
├─FCOS
│  └─data
│      └─coco
│        └─coco2017
│      └─VOCdevkit
│        └─VOC2007
│        └─VOC2012
```

Pascal scripts

```bash
cd FCOS
mkdir data
# if data is not existed.
cd data
# download
wget http://pjreddie.com/media/files/VOCtrainval_06-Nov-2007.tar
wget http://pjreddie.com/media/files/VOCtest_06-Nov-2007.tar
wget http://pjreddie.com/media/files/VOCdevkit_08-Jun-2007.tar
# tar
tar -xvf VOCdevkit_08-Jun-2007.tar
tar -xvf VOCtest_06-Nov-2007.tar
tar -xvf VOCtrainval_06-Nov-2007.tar
# Delete 
rm -rf VOCdevkit_08-Jun-2007.tar
rm -rf VOCtest_06-Nov-2007.tar
rm -rf VOCtrainval_06-Nov-2007.tar
```

### Train

1. set model/config.py
2. run train_voc.py/train_coco.py

```bash
# eg: Pascal VOC 2006
python train_voc.py --batch_size 8 --accumulate_step 4 --LR_INIT 0.001 --height 400 --width 667 --interval 1 --saved_path './checkpoint/base' --epochs 50
```

3. 400 * 677 need about 9.4G GPU

### Eval

```bash
python eval_voc.py --height 400 --width 677 --checkpoint  './checkpoint/base/model_8_400_667_50.pth' --strict 0
```

### Thanks

Thanks to [@ZhangZhengHao](https://github.com/zhenghao977),  I referenced some codes.