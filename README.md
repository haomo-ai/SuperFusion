# SuperFusion

This repository contains the implementation of the paper:

**SuperFusion: Multilevel LiDAR-Camera Fusion for Long-Range HD Map Generation**

[Hao Dong](https://sites.google.com/view/dong-hao/), Xianjing Zhang, Jintao Xu, Rui Ai, Weihao Gu, Huimin Lu, [Juho Kannala](https://users.aalto.fi/~kannalj1/) and [Xieyuanli Chen](http://xieyuanli-chen.com/) 

[Link](https://arxiv.org/abs/2211.15656) to the arXiv version of the paper is available.

<img src="pics/overview.png" width="800">

Pipeline overview of SuperFusion. Our method fuses camera and LiDAR data in three levels: the data-level fusion fuses depth information from LiDAR to improve the accuracy of image depth estimation, the feature-level fusion uses cross-attention for long-range LiDAR BEV feature prediction with the guidance of image features, and the BEV-level fusion aligns two branches to generate high-quality fused BEV features. Finally, the fused BEV features can support different heads, including semantic segmentation, instance embedding, and direction prediction, and finally post-processed to generate the HD map prediction.

![SuperFusion_demo](pics/SuperFusion_demo.gif)

## Abstract
High-definition (HD) semantic map generation of the environment is an essential component of autonomous driving. Existing methods have achieved good performance in this task by fusing different sensor modalities, such as LiDAR and camera. However, current works are based on raw data or network feature-level fusion and only consider short-range HD map generation, limiting their deployment to realistic autonomous driving applications. In this paper, we focus on the task of building the HD maps in both short ranges, i.e., within 30 m, and also predicting long-range HD maps up to 90 m, which is required by downstream path planning and control tasks to improve the smoothness and safety of autonomous driving. To this end, we propose a novel network named SuperFusion, exploiting the fusion of LiDAR and camera data at multiple levels. We use LiDAR depth to improve image depth estimation and use image features to guide long-range LiDAR feature prediction. We benchmark our SuperFusion on the nuScenes dataset and a self-recorded dataset and show that it outperforms the state-of-the-art baseline methods with large margins on all intervals. Additionally, we apply the generated HD map to a downstream path planning task, demonstrating that the long-range HD maps predicted by our method can lead to better path planning for autonomous vehicles.

## Code
### Prepare
1. Download [nuScenes dataset](https://www.nuscenes.org/) (Full dataset and Map expansion) and unzip files. The folder should be like
<img src="pics/dataset.png" width="200">

2. Install dependencies by running
```
pip install -r requirement.txt
```

3. Download the pretrained [DeepLabV3 model](https://download.pytorch.org/models/deeplabv3_resnet101_coco-586e9e4e.pth) and place within the `checkpoints` directory
   
4. Download the pretrained [SuperFusion model](https://drive.google.com/file/d/1UTgughJ71Rn0zPUDTXFo__HJS-57lwNG/view?usp=sharing) and place within the `runs` directory
   
### Training
```
python train.py --instance_seg --direction_pred --depth_sup --dataroot /path/to/nuScenes/ --pretrained --add_depth_channel
```

### Evaluation
1. Evaluate IoU
```
python evaluate_iou_split.py --dataroot /path/to/nuScenes/ --modelf runs/model.pt --instance_seg --direction_pred --depth_sup --add_depth_channel --pretrained
```
2. Evaluate CD and AP
```
python export_pred_to_json.py --dataroot /path/to/nuScenes/ --modelf runs/model.pt --depth_sup --add_depth_channel --pretrained
```
```
python evaluate_json_split.py --result_path output.json --dataroot /path/to/nuScenes/
```
### Visualization
```
python vis_prediction_gt.py --instance_seg --direction_pred --dataroot /path/to/nuScenes/
```
```
python vis_prediction.py --modelf runs/model.pt --instance_seg --direction_pred --depth_sup --pretrained --add_depth_channel --version v1.0-trainval --dataroot /path/to/nuScenes/
```
## Long-range HD map generation on nuScenes dataset
<img src="pics/results.png" width="800">

## Instance detection results on nuScenes dataset.
<img src="pics/nuScenes_ap.png" width="800">

## More examples of self-recorded dataset
<img src="pics/haomo_data.png" width="500">

## Long-range HD map generation on self-recorded dataset
<img src="pics/haomo_results.png" width="800">

## Instance detection results on self-recorded dataset.
<img src="pics/haomo_ap.png" width="800">

## Citation
If you use our implementation in your academic work, please cite the corresponding [paper](https://arxiv.org/abs/2211.15656):
```
@article{dong2022SuperFusion,
	author   = {Hao Dong and Xianjing Zhang and Jintao Xu and Rui Ai and Weihao Gu and Huimin Lu and Juho Kannala and Xieyuanli Chen},
	title    = {{SuperFusion: Multilevel LiDAR-Camera Fusion for Long-Range HD Map Generation}},
	journal  = {arXiv preprint arXiv:2211.15656},
	year     = {2022},
}
```

## Acknowledgement

Many thanks to the excellent open-source projects [HDMapNet](https://github.com/Tsinghua-MARS-Lab/HDMapNet),[LSS](https://github.com/nv-tlabs/lift-splat-shoot),[AlignSeg](https://github.com/speedinghzl/AlignSeg) and [CaDDN](https://github.com/TRAILab/CaDDN).