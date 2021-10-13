# ISE-RCNN
ISE-RCNN: Image Semantics Enhancement Network for Robust 3D Object Detection. This repo is based on [OpenPCDet-VoxelRCNN](https://arxiv.org/abs/2012.15712) baseline, you can modify the baseline as you like.

# Installation
Our codes are tested in the following environment:
- Linux
- Python 3.7
- PyTorch 1.8.0
- CUDA CUDA 11.1
- [`spconv v1.2`](https://github.com/traveller59/spconv)

Please install pytorch and spconv before other requirements, references for installing are as follows:

1. Install pytorch. See https://pytorch.org/get-started/previous-versions/ for more details. Skip this if you already install pytorch properlly.

```
$ conda install pytorch==1.7.1 torchvision==0.8.2 cudatoolkit=11.0 -c pytorch
```
2. Clone our repo.
```
$ git clone https://github.com/JeffWang987/ISE-RCNN
```
3. Instll spconv, see https://github.com/traveller59/spconv for more details. A possible installing scripts are as follows:
```
$ cd ./ISE-RCNN/3rdparty
$ git clone https://github.com/traveller59/spconv.git --recursive
$ sudo apt-get install libboost-all-dev
$ cd ./ISE-RCNN/3rdparty/spconv
$ python setup.py bdist_wheel (make sure cmake>=3.13.2)
$ cd ./ISE-RCNN/3rdparty/spconv/dist
$ pip install xxx.whl (install generated whl file)

```
4. Install other requirements.
```
$ cd ./ISE-RCNN
$ pip install -r requirements.txt
$ python setup.py develop

```

# Data preparation
## KITTI
1. Similar to OpenPCDet, please organize the raw data as follows:
```
ISE-RCNN
├── data
│   ├── kitti
│   │   │── ImageSets
│   │   │── training
│   │   │   ├──calib & velodyne & label_2 & image_2 & (optional: planes) & (optional: depth_2)
│   │   │── testing
│   │   │   ├──calib & velodyne & image_2
├── pcdet
├── tools
```

2. Generate the data infos
Note if you have already generated the data infos by OpenPCDet, you dont have run the following command.
```
$ python -m pcdet.datasets.kitti.kitti_dataset create_kitti_infos tools/cfgs/dataset_configs/kitti_dataset.yaml
```
3. The next command must be implemented.
```
$ python -m pcdet.datasets.kitti.kitti_dataset create_kitti_pts_imgs_dbs tools/cfgs/dataset_configs/kitti_dataset.yaml
```

# Trian and Eval
1. Train or test on multi GPUs.
```
$ sh scripts/dist_train.sh ${NUM_GPUS} --cfg_file ./ISE-RCNN/tools/cfgs/kitti_models/ISE-RCNN.yaml
$ sh scripts/dist_test.sh ${NUM_GPUS}  --cfg_file ./ISE-RCNN/tools/cfgs/kitti_models/ISE-RCNN.yaml
```
2. Train or test on single GPUs.
```
$ python train.py --cfg_file ./ISE-RCNN/tools/cfgs/kitti_models/ISE-RCNN.yaml
$ python test.py --cfg_file ./ISE-RCNN/tools/cfgs/kitti_models/ISE-RCNN.yaml
```

# Submission Link on KITTI: [ISE-RCNN results on KITTI](http://www.cvlibs.net/datasets/kitti/eval_object_detail.php?&result=71000703378db66e09cd6a3ab44a37f2e69fd02b)
Until Sep 04 2021, our method rank 1st on Pedestrain and Cyclist leaderboards(compared with published works)

# Acknowledgement
We thank the codebase [OpenPCDet](https://github.com/open-mmlab/OpenPCDet) and [`spconv`](https://github.com/traveller59/spconv), our work is based on them.

We thank the great work [CLIP](https://github.com/openai/CLIP), which inspires our work.
