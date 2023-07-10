
![title1](https://github.com/JulesK8/Vehicle-Detection/assets/134627326/6d96c76e-6f42-4eba-9eb3-9fa88e55fbab)

# Vehicle-Detection


A curated list of existing methods and datasets for vehicle detection in an autonomous driving context.  
You are welcome to update the list.  
Author: Jules KARANGWA  
Contact: karangwa@mail.ustc.edu.cn  

> **Note:**   
> A review paper related to this list is accepted to be published in IEEE Transactions.  
The paper is currently under final production stage. I will keep updating this repo for the
latest works in vehicle detection field, and i will provide the link for the paper after publication.   
If you find that the paper contents are useful, please cite this paper in your work.  

# Intro  
Efficient and accurate vehicle detection is one of the essential tasks in the environment  
perception of an autonomous vehicle. Therefore, numerous  algorithms for vehicle detection  
have been developed. 

# Datasets  
A more detailed table about datasets used in vehicle detection can be found on page 4 of my  
review paper.  
 
The table below summarize some of the most known datasets for vehicle detection include KITTI datasets, nuScene,  
Waymo open datasets, ApolloScape, CityScape, Oxford radar robot car, H3D, Lyft level 5, Pandasets, ONCE (one million scenes),   
AIO Drive, Ford multi-AV seasonal dataset, DDAD, CADC (canadian adverse driving conditions), A2D2, A*3D dataset, Argoverse,  
BLVD (building a large-scale 5D semantics benchmark for autonomous driving), Rope3D, Itacha365.  

**Some of Multisensor-based datasets for vehicle detection**
| Datasets | Scene | Classes | Annotations | 3D boxes | Sensors | Place | Website |
|----------|-------|---------|---------|-------|---------|---------|---------|
| KITTI    |   22  |    8    | 15k | 200k |   CL   | Germany | [KITTI](https://www.cvlibs.net/datasets/kitti/), [pdf](https://www.cvlibs.net/publications/Geiger2013IJRR.pdf)|
|nuScene   |1k     | 23      | 40k | 1.4M | CL    |Boston & Singapore| [nuScene](https://www.nuscenes.org/), [pdf](https://arxiv.org/abs/1903.11027) |
|Waymo Open| 1k    | 4       | 200k | 12M | CL | Different cities| [WOD](https://waymo.com/open/), [pdf](https://arxiv.org/pdf/1912.04838.pdf)|
| Apollo scape| - | 35 | 130k| 70k | CL | 10 cities in China | [Apollo](https://apolloscape.auto/), [pdf](https://arxiv.org/pdf/1803.06184.pdf)|
| City Scape| - | 30| 30k| -|CL| 50 cities|[CityScape](https://www.cityscapes-dataset.com/), [pdf](https://arxiv.org/abs/1604.01685)|
|Oxford Robot Car| -| - | - | -|CLR| Oxford UK |[ORC](https://robotcar-dataset.robots.ox.ac.uk/), [pdf](https://robotcar-dataset.robots.ox.ac.uk/images/robotcar_ijrr.pdf)|
|H3D|160|8|27K|1.07M|L|San Francisco Bay|[h3d](https://usa.honda-ri.com/h3d), [pdf](https://arxiv.org/abs/1903.01568) |
|Lyft level 5| -|-|-|-|CL| Palo Alto, California|[LL5](https://woven-planet.github.io/l5kit/dataset.html), [pdf](https://arxiv.org/pdf/2006.14480.pdf)| 
|Panda sets| 8|28|80k|-|CL|-|[Panda](https://scale.com/open-av-datasets/pandaset), [pdf](https://arxiv.org/abs/2112.12610)|
|ONCE|1M|5|15k|-|CL|china|[once](https://once-for-auto-driving.github.io/), [pdf](https://arxiv.org/abs/2106.11037)|
|AIODrive|-|-|-|10M|CLR|-|[AIO](http://www.aiodrive.org/download.html), [pdf](https://www.xinshuoweng.com/papers/AIODrive/arXiv.pdf)|
|Ford Multi-AV| -|-|-|-|CL|Michigan USA|[Ford](https://avdata.ford.com/), [pdf](https://arxiv.org/abs/2003.07969)|
|DDAD|-|-|-|-|CL|USA & Japan|[ddad](https://github.com/TRI-ML/DDAD), [pdf](https://arxiv.org/pdf/2103.16690.pdf)|
|CADC|75|10|7k|-|CL|Waterloo Canada| [cadc](http://cadcd.uwaterloo.ca/), [pdf](https://arxiv.org/abs/2001.10117)|
|A2D2|-|-|12k|-|CL|Germany|[audi](https://www.a2d2.audi/a2d2/en/download.html), [pdf](https://arxiv.org/abs/2004.06320)|
|Ax3D|-|7|230k|-|CL|Singapore|[a3d](https://www.v7labs.com/open-datasets/a-3d), [pdf](https://arxiv.org/abs/1909.07541)|
|Argoverse|1K|-|-|-|CL|Pittdburg & Miami USA|[arg2](https://argoverse.org/av2.html), [arg1](https://www.argoverse.org/av1.html), [pdf](https://arxiv.org/abs/1911.02620)|
|BLVD|654|3|249k|-|CL|Changshu China|[blvd](https://github.com/VCCIV/BLVD), [pdf](https://arxiv.org/abs/1903.06405)|
|Itacha 365|40|6|-|-|CL|Itacha, New York|[Itacha](https://ithaca365.mae.cornell.edu/), [pdf](https://arxiv.org/abs/2208.01166)|
|Rope3D|26|12|-|1.5M|CL|-|[rope3d]( https://thudair.baai.ac.cn/rope), [pdf](https://arxiv.org/abs/2203.13608)|

> **Note:**  
> C, L, and R stands for Camera, LiDAR and Radar respectively.
> Note that these datasets are not only used to detect vehicles, but also are used to detect other objects
> such as Pedestrians and Cyclists.


**4D radar datasets for vehicle detection**
|Dataset|Radar Type|Data type| Annotation| Reference|
|-|-|-|-|-|
|Astyx Hires2019|Astyx 6455 HiRes Middel Range|PC|3D bbox|[Astyx](https://github.com/under-the-radar/radar_dataset_astyx)|
|TJ4DRadSet|Oculii Eagle Long Range|	PC|3D bbox, TrackID|[TJ4DRadSet](https://github.com/TJRadarLab/TJ4DRadSet)|
|K-Radar|Macnica RETINA|RAD|3D bbox, Track ID|[K-Radar](https://github.com/kaist-avelab/K-Radar)|
|RADIal|Valeo Middel Range DDM|ADC,RAD,PC|Point-level Vehicle; Open Space Mask|[Radial](https://github.com/valeoai/RADIal)|
|View-of-Delft|ZF FRGen21 Short Range|PC|3D bbox|[VOD](https://tudelft-iv.github.io/view-of-delft-dataset/)|

> **Hints:**     
> Astyx is small, VoD focuses on VRU classification, RADIal's annotation is coarse but provides raw data, TJ4D features    
> for its long range detection, K-Radar provides RAD tensor and 3D annotations. TJ4D and K-radar are not yet public available.

# The recent vehicle detection methods
## Machine Vision-Based Vehicle Detection
According to the principles of the existing algorithm, machine vision-based vehicle detection can be categorized as traditional-based,  
machine-learning-based, and deep-learning-based methods.  

**1. Traditional methods**  
![figure 1](https://github.com/JulesK8/Vehicle-Detection/assets/134627326/1034cb8c-4514-4ab4-bc17-9ca7888d5848)

***Appearance-based methods***  

**a. Symmetry-based methods**  
* Multipart vehicle detection using symmetry-derived analysis and active learning [paper](https://ieeexplore.ieee.org/abstract/document/7368153/)
* Symmetry-based monocular vehicle detection system [paper](https://link.springer.com/article/10.1007/s00138-011-0355-7) 
* Video vehicle detection algorithm based on edge symmetry [paper](https://bhxb.buaa.edu.cn/bhzk/en/article/id/9405)  

**b. Edge feature-based methods**
* Edge-based forward vehicle detection method for complex scenes [paper](https://ieeexplore.ieee.org/abstract/document/6904044/)  
* Multiscale edge fusion for vehicle detection based on difference of Gaussian [paper](https://www.sciencedirect.com/science/article/pii/S0030402616000292)  
* Vehicle  detection based on underneath vehicle shadow using edge features [paper](https://ieeexplore.ieee.org/abstract/document/7893608/)  
* A feature-based recognition scheme for traffic scenes [paper](https://ieeexplore.ieee.org/abstract/document/528285/)  

**c. Color-based methods**
* Color modeling by spherical influence field in sensing driving environment [paper](https://ieeexplore.ieee.org/abstract/document/898350/)  
* Daytime preceding vehicle brake light detection using monocular vision [paper](https://ieeexplore.ieee.org/abstract/document/7247631)  
* Vehicle detection based on color analysis [paper](https://ieeexplore.ieee.org/abstract/document/6380975/)  
* Road detection and vehicles tracking by vision for an on-board acc system in the velac vehicle [paper](https://ieeexplore.ieee.org/abstract/document/859833)

**d. Taillight-based methods**
* Rear-lamp vehicle detection and tracking in low-exposure color video for night conditions [paper](https://ieeexplore.ieee.org/abstract/document/5446402/)  
* Vision-based nighttime vehicle detection using CenSurE and SVM [paper](https://ieeexplore.ieee.org/abstract/document/7103307)  
* An improved technique for Night-time Vehicle detection [paper](https://ieeexplore.ieee.org/abstract/document/8554712)  
* Looking at vehicles in the night: Detection and dynamics of rear lights [paper](https://ieeexplore.ieee.org/abstract/document/7750549)  
* Vehicle detection using tail light segmentation [paper](https://ieeexplore.ieee.org/abstract/document/6021126)  

**e. Underneath shadow-based methods**
* Vehicle detection based on underneath vehicle shadow using edge features [paper](https://ieeexplore.ieee.org/abstract/document/7893608/)  
* Shadow-based vehicle detection in urban traffic [paper](https://www.mdpi.com/193936)  
* Shadow Based On-Road Vehicle Detection and Verification Using HAAR Wavelet Packet Transform [paper](https://ieeexplore.ieee.org/abstract/document/1598621/)  
* Shadow detection in camera-based vehicle detection: survey and analysis [paper](https://www.spiedigitallibrary.org/journals/Journal-of-Electronic-Imaging/volume-25/issue-5/051205/Shadow-detection-in-camera-based-vehicle-detection--survey-and/10.1117/1.JEI.25.5.051205.short?SSO=1)  

**f. Texture-based methods**
* Robust vehicle detection in vision systems based on fast wavelet transform and texture analysis [paper](https://ieeexplore.ieee.org/abstract/document/4339088/)  
* Real-time small obstacle detection on highways using compressive RBM road reconstruction [paper](https://ieeexplore.ieee.org/abstract/document/7225680/) 

***Stereo-vision-based methods***    
* GOLD: A parallel real-time stereo vision system for generic obstacle and lane detection [paper](https://ieeexplore.ieee.org/abstract/document/650851/)  
* Vehicle detection based on multifeature extraction and recognition adopting RBF neural network on ADAS system [paper](https://www.hindawi.com/journals/complexity/2020/8842297/)  

***Motion-based methods***    
* Motion based vehicle detection on motorways [paper](https://ieeexplore.ieee.org/abstract/document/528329/)  
* Motion-based vehicle detection in Hsuehshan Tunnel [paper](https://ieeexplore.ieee.org/abstract/document/7449856/) 
* Moving vehicle detection and tracking in unstructured environments [paper](https://ieeexplore.ieee.org/abstract/document/6224636/)  
* Go with the flow: Improving Multi-View vehicle detection with motion cues [paper](https://ieeexplore.ieee.org/abstract/document/6977422/)  

**2. Machine learning-based methods**  

**a. HOG:**   
* Histograms of oriented gradients [paper](https://courses.cs.duke.edu/fall17/compsci527/notes/hog.pdf)  
* Real-time vehicle detection using histograms of oriented gradients and AdaBoost classification [paper](https://www.sciencedirect.com/science/article/pii/S0030402616305423)  
* Front and rear vehicle detection using hypothesis generation and verification [paper](https://www.academia.edu/download/37934652/4413sipij03.pdf)  
* Image-based on-road vehicle detection using cost-effective histograms of oriented gradients [paper](https://www.sciencedirect.com/science/article/abs/pii/S1047320313001478)  
* Vision-based vehicle detection system with consideration of the detecting location [paper](https://ieeexplore.ieee.org/abstract/document/6175131/)  

**b. Haar-like feature:**  
* Rapid object detection using a boosted cascade of simple features [paper](https://ieeexplore.ieee.org/abstract/document/990517/)  
* Efficient feature selection and classification for vehicle detection [paper](https://ieeexplore.ieee.org/abstract/document/6898836/)  
* A general active-learning framework for on-road vehicle recognition and tracking [paper](https://ieeexplore.ieee.org/abstract/document/5411825)  
* A vehicle detection system based on haar and triangle features [paper](https://ieeexplore.ieee.org/abstract/document/5164288/)  
* Monocular precrash vehicle detection: features and classifiers [paper](https://ieeexplore.ieee.org/abstract/document/1643708/)  
* SURF: Speed Up Robust Features [paper](https://lirias.kuleuven.be/73068?limo=0)  
* Integrating appearance and edge features for sedan vehicle detection in the blind-spot area [paper](https://ieeexplore.ieee.org/abstract/document/6145682/)  
* Object recognition from local scale-invariant feature [paper](https://ieeexplore.ieee.org/abstract/document/790410/)    
* Vehicle detection using an extended hidden random field mode [paper](https://ieeexplore.ieee.org/abstract/document/6083135/)  

**3. Deep-learning-based methods for vehicle detection**  

**_A. Object detection-based methods_**  


**1. Anchor-based detectors**  
**a. Two-stage detection networks**  
* Rich Feature Hierarchies for Accurate Object Detection and Semantic Segmentation [RCNN](http://openaccess.thecvf.com/content_cvpr_2014/html/Girshick_Rich_Feature_Hierarchies_2014_CVPR_paper.html)  
* Fast R-CNN [Fast RCNN](http://openaccess.thecvf.com/content_iccv_2015/html/Girshick_Fast_R-CNN_ICCV_2015_paper.html)  
* Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks [Faster RCNN](https://proceedings.neurips.cc/paper/2015/hash/14bfa6bb14875e45bba028a21ed38046-Abstract.html)  
* Spatial Pyramid Pooling in Deep Convolutional Networks for Visual Recognition [SPP-Net](https://ieeexplore.ieee.org/abstract/document/7005506/)

![two-stage object detection network](https://github.com/JulesK8/Vehicle-Detection/assets/134627326/d603159c-be77-465b-af72-65eaf8ecb244)


> Note:  
> These are some pioneering networks in context of object detection, i.e, these networks are not only used to detect  
> vehicles but also detect other objects. Consider taking a look into my review paper for a deep understanding.

**b. One-stage detection networks**  

* You Only Look Once: Unified, Real-Time Object Detection [YOLO v1](https://www.cv-foundation.org/openaccess/content_cvpr_2016/html/Redmon_You_Only_Look_CVPR_2016_paper.html)   
* SSD: Single Shot MultiBox Detector [SSD](https://link.springer.com/chapter/10.1007/978-3-319-46448-0_2) 
* YOLO9000: Better, Faster, Stronger [YOLO v2](http://openaccess.thecvf.com/content_cvpr_2017/html/Redmon_YOLO9000_Better_Faster_CVPR_2017_paper.html)    
* YOLOv3: An Incremental Improvement [YOLO v3](https://arxiv.org/abs/1804.02767)    
* M2Det: A Single-Shot Object Detector Based on Multi-Level Feature Pyramid Network [M2Det](https://ojs.aaai.org/index.php/AAAI/article/view/4962)  
* YOLOv4: Optimal Speed and Accuracy of Object Detection [YOLO v4](https://arxiv.org/abs/2004.10934)    
* YOLOv7: Trainable Bag-of-Freebies Sets New State-of-the-Art for Real-Time Object Detectors [YOLO v7](http://openaccess.thecvf.com/content/CVPR2023/html/Wang_YOLOv7_Trainable_Bag-of-Freebies_Sets_New_State-of-the-Art_for_Real-Time_Object_Detectors_CVPR_2023_paper.html)  

![one-stage object detecton network](https://github.com/JulesK8/Vehicle-Detection/assets/134627326/00f2cdda-992d-49ba-8e72-bb24739fc8bd)



> Note:  
> These are some pioneering networks in context of object detection, i.e, these networks are not only used to detect  
> vehicles but also detect other objects. Consider taking a look into my review paper for a deep understanding.  

**2. Anchor-free detectors**  
The existing studies show that anchor-free models can be classified as key-point- and center-based. Key-point-based  
methods detect specific object points, such as center and corner points, and group them for bounding box prediction.   
whereas, center-based methods directly predict the object's center point and perform the object bounding box regression.  

**a. Anchor-free keypoint-based**  
* Bottom-Up Object Detection by Grouping Extreme and Center Points [ExtremeNet](http://openaccess.thecvf.com/content_CVPR_2019/html/Zhou_Bottom-Up_Object_Detection_by_Grouping_Extreme_and_Center_Points_CVPR_2019_paper.html)  
* CornerNet: Detecting Objects as Paired Keypoints [CornerNet](http://openaccess.thecvf.com/content_ECCV_2018/html/Hei_Law_CornerNet_Detecting_Objects_ECCV_2018_paper.html)  
* Objects as Points [CenterNet-HG](https://arxiv.org/abs/1904.07850)  
* Grid R-CNN [Grid RCNN](http://openaccess.thecvf.com/content_CVPR_2019/html/Lu_Grid_R-CNN_CVPR_2019_paper.html)
* CornerNet-Lite: Efficient Keypoint Based Object Detection [CornerNet-Lite](https://arxiv.org/abs/1904.08900)
* CenterNet: Keypoint Triplets for Object Detection [CenterNet](http://openaccess.thecvf.com/content_ICCV_2019/html/Duan_CenterNet_Keypoint_Triplets_for_Object_Detection_ICCV_2019_paper.html)
* RepPoints: Point Set Representation for Object Detection [RepPoints](http://openaccess.thecvf.com/content_ICCV_2019/html/Yang_RepPoints_Point_Set_Representation_for_Object_Detection_ICCV_2019_paper.html)

**b. Anchor-free center-based**  
* FCOS: Fully Convolutional One-Stage Object Detection [FCOS](https://openaccess.thecvf.com/content_ICCV_2019/html/Tian_FCOS_Fully_Convolutional_One-Stage_Object_Detection_ICCV_2019_paper.html)  
* FoveaBox: Beyound Anchor-Based Object Detection [Foveabox](https://ieeexplore.ieee.org/abstract/document/9123553/)
* Feature Selective Anchor-Free Module for Single-Shot Object Detection [FSAF](http://openaccess.thecvf.com/content_CVPR_2019/html/Zhu_Feature_Selective_Anchor-Free_Module_for_Single-Shot_Object_Detection_CVPR_2019_paper.html)
* Region Proposal by Guided Anchoring [GA-RPN](http://openaccess.thecvf.com/content_CVPR_2019/html/Wang_Region_Proposal_by_Guided_Anchoring_CVPR_2019_paper.html)
* Bridging the Gap Between Anchor-Based and Anchor-Free Detection via Adaptive Training Sample Selection [ATSS](https://openaccess.thecvf.com/content_CVPR_2020/html/Zhang_Bridging_the_Gap_Between_Anchor-Based_and_Anchor-Free_Detection_via_Adaptive_CVPR_2020_paper.html)  

**c. Transformer-based detector**
* Dynamic graph transformer for 3D object detection [DGT-Det3D](https://www.sciencedirect.com/science/article/abs/pii/S0950705122011819)  
* Efficient Transformer-based 3D Object Detection with Dynamic Token Halting [ET-DTH](https://arxiv.org/abs/2303.05078)  
* Object Detection Based on Swin Deformable Transformer-BiPAFPN-YOLOX [RDSA](https://www.hindawi.com/journals/cin/2023/4228610/)  
* Voxel Set Transformer: A Set-to-Set Approach to 3D Object Detection From Point Clouds [VoxSet](https://openaccess.thecvf.com/content/CVPR2022/html/He_Voxel_Set_Transformer_A_Set-to-Set_Approach_to_3D_Object_Detection_CVPR_2022_paper.html)  
* VISTA: Boosting 3D Object Detection via Dual Cross-VIew SpaTial Attention [VISTA](http://openaccess.thecvf.com/content/CVPR2022/html/Deng_VISTA_Boosting_3D_Object_Detection_via_Dual_Cross-VIew_SpaTial_Attention_CVPR_2022_paper.html)  
* CenterFormer: Center-Based Transformer for 3D Object Detection [CenterFormer](https://link.springer.com/chapter/10.1007/978-3-031-19839-7_29)  
* SWFormer: Sparse Window Transformer for 3D Object Detection in Point Clouds [SW Former](https://link.springer.com/chapter/10.1007/978-3-031-20080-9_25)  
* Swin transformer based vehicle detection in undisciplined traffic environment [STVD](https://www.sciencedirect.com/science/article/pii/S0957417422020103)  
* Voxel Transformer for 3D Object Detection [VoTr](http://openaccess.thecvf.com/content/ICCV2021/html/Mao_Voxel_Transformer_for_3D_Object_Detection_ICCV_2021_paper.html)  
* Multi-Source Features Fusion Single Stage 3D Object Detection With Transformer [MFT-SST](https://ieeexplore.ieee.org/abstract/document/10042064/)  

> Note:  
> The above models are inspired by the original work presented as [Vit](https://arxiv.org/abs/2010.11929).  

**_B. Semantic segmentation-based detection networks_**  
Semantic segmentation associates a label or a category with every pixel in the image.  

>Note:  
>It can be mentioned that most of semantic segmentation methods are applicable in the field of vehicle detection.

* Rethinking the Inception Architecture for Computer Vision [DeepMask](https://www.cv-foundation.org/openaccess/content_cvpr_2016/html/Szegedy_Rethinking_the_Inception_CVPR_2016_paper.html)  
* DeepLab: Semantic Image Segmentation with Deep Convolutional Nets, Atrous Convolution, and Fully Connected CRFs [DeepLab](https://ieeexplore.ieee.org/abstract/document/7913730/)  
* DeepLab2: A TensorFlow Library for Deep Labeling [DeepLab2](https://arxiv.org/abs/2106.09748)  
* Rethinking Atrous Convolution for Semantic Image Segmentation [DeepLab3](https://arxiv.org/abs/1706.05587)  
* Pyramid Scene Parsing Network [PSPNet](http://openaccess.thecvf.com/content_cvpr_2017/html/Zhao_Pyramid_Scene_Parsing_CVPR_2017_paper.html)  
* ICNet for Real-Time Semantic Segmentation on High-Resolution Images [ICNet](http://openaccess.thecvf.com/content_ECCV_2018/html/Hengshuang_Zhao_ICNet_for_Real-Time_ECCV_2018_paper.html)  
* RefineNet: Multi-Path Refinement Networks for High-Resolution Semantic Segmentation [RefineNet](http://openaccess.thecvf.com/content_cvpr_2017/html/Lin_RefineNet_Multi-Path_Refinement_CVPR_2017_paper.html)  
* Semi Supervised Semantic Segmentation Using GAN [paper](https://openaccess.thecvf.com/content_iccv_2017/html/Souly__Semi_Supervised_ICCV_2017_paper.html)  
* SegNet: A Deep Convolutional Encoder-Decoder Architecture for Image Segmentation [SegNet](https://ieeexplore.ieee.org/abstract/document/7803544/)  
* BiSeNet: Bilateral Segmentation Network for Real-time Semantic Segmentation [BiseNet](http://openaccess.thecvf.com/content_ECCV_2018/html/Changqian_Yu_BiSeNet_Bilateral_Segmentation_ECCV_2018_paper.html)  
* DFANet: Deep Feature Aggregation for Real-Time Semantic Segmentation [DFANet](http://openaccess.thecvf.com/content_CVPR_2019/html/Li_DFANet_Deep_Feature_Aggregation_for_Real-Time_Semantic_Segmentation_CVPR_2019_paper.html)  
* ENet: A Deep Neural Network Architecture for Real-Time Semantic Segmentation [Enet](https://arxiv.org/abs/1606.02147)  
* Lednet: A Lightweight Encoder-Decoder Network for Real-Time Semantic Segmentation [LedNet](https://ieeexplore.ieee.org/abstract/document/8803154/)  
* ESPnet: End-to-End Speech Processing Toolkit [EspNet](https://arxiv.org/abs/1804.00015)  
* CGNet: A Light-Weight Context Guided Network for Semantic Segmentation [CGNet](https://ieeexplore.ieee.org/abstract/document/9292449)  

## **LiDAR-based methods for vehicle detection**  

LiDAR is a sensor that perceives the environment by transmitting laser signals and receiving the reflected signals   
from the obstacles in the environment.  

Practically, existing LiDAR-based methods can be classified as traditional and deep learning techniques.  

**1. Traditional methods**  

* A real-time grid map generation and object classification for ground-based 3D LIDAR data using image analysis techniques [paper](https://ieeexplore.ieee.org/abstract/document/5651197/) 
* A Graph-Based Method for Joint Instance Segmentation of Point Clouds and Image Sequences [paper](https://ieeexplore.ieee.org/abstract/document/9560765)  
* Range image-based density-based spatial clustering of application with noise clustering method of three-dimensional point clouds [paper](https://journals.sagepub.com/doi/pdf/10.1177/1729881418762302)  
* Improving LiDAR point cloud classification using intensities and multiple echoes [paper](https://ieeexplore.ieee.org/abstract/document/7354098)  
* Drivable road detection with 3D point clouds based on the MRF for intelligent vehicle [paper](https://link.springer.com/chapter/10.1007/978-3-319-07488-7_4)  

**2. LiDAR deep-learning-based method**  
Considering the data type where we extract feature maps, LiDAR deep learning methods can be categorized as  
raw data-based, projection-based, and voxel-based.

![categories of different lidar deep-learning based method](https://github.com/JulesK8/Vehicle-Detection/assets/134627326/9958d247-6233-43eb-af46-062cab8d4923)

**Raw data based**
* 2017 PointNet: Deep Learning on Point Sets for 3D Classification and Segmentation [paper](https://openaccess.thecvf.com/content_cvpr_2017/papers/Qi_PointNet_Deep_Learning_CVPR_2017_paper.pdf), [code](https://github.com/charlesq34/pointnet)  
* 2017 PointNet++: Deep Hierarchical Feature Learning on Point Sets in a Metric Space [paper](https://proceedings.neurips.cc/paper/2017/file/d8bf84be3800d12f74d8b05e9b89836f-Paper.pdf), [code](https://github.com/charlesq34/pointnet2)  
* 2018 IPOD: Intensive Point-based Object Detector for Point Cloud [paper](https://arxiv.org/pdf/1812.05276.pdf)   
* 2018 RoarNet: A Robust 3D Object Detection based on RegiOn Approximation Refinement [paper](https://arxiv.org/pdf/1811.03818.pdf), [code](https://github.com/collector-m/RoarNet)  
* 2019 PointRCNN: 3D Object Proposal Generation and Detection from Point Cloud [paper](https://arxiv.org/pdf/1812.04244.pdf), [code](https://github.com/sshaoshuai/PointRCNN)  
* 2019 STD: Sparse-to-Dense 3D Object Detector for Point Cloud [paper](https://arxiv.org/pdf/1907.10471.pdf). 
* 2019 KPConv: Flexible and Deformable Convolution for Point Clouds [paper](https://arxiv.org/pdf/1904.08889.pdf), [code](https://github.com/HuguesTHOMAS/KPConv)  
* 2019 ShellNet: Efficient Point Cloud Convolutional Neural Networks using Concentric Shells Statistics [paper](https://arxiv.org/pdf/1908.06295.pdf), [code](https://github.com/hkust-vgd/shellnet)      
* 2020 SalsaNet: Fast Road and Vehicle Segmentation in LiDAR Point Clouds for Autonomous Driving [paper](https://arxiv.org/pdf/1909.08291.pdf), [code](https://gitlab.com/aksoyeren/salsanet)  
* 2020 RandLA-Net: Efficient Semantic Segmentation of Large-Scale Point Clouds [paper](https://arxiv.org/pdf/1911.11236.pdf), [code](https://github.com/QingyongHu/RandLA-Net)  
* 2020 From Points to Parts: 3D Object Detection from Point Cloud with Part-aware and Part-aggregation Network [paper](https://arxiv.org/pdf/1907.03670.pdf), [code](https://github.com/open-mmlab/OpenPCDet)  
* 2020 3DSSD: Point-based 3D Single Stage Object Detector [paper](https://arxiv.org/pdf/2002.10187.pdf), [code](https://github.com/dvlab-research/3DSSD)  
* 2020 SASSD: Structure Aware Single-stage 3D Object Detection from Point Cloud [paper](https://www4.comp.polyu.edu.hk/~cslzhang/paper/SA-SSD.pdf), [code](https://github.com/skyhehe123/SA-SSD)  
* 2020 CIA-SSD: Confident IoU-Aware Single-Stage Object Detector From Point Cloud [paper](https://arxiv.org/pdf/2012.03015.pdf), [code](https://github.com/Vegeta2020/CIA-SSD)   
* 2020 Point-GNN: Graph Neural Network for 3D Object Detection in a Point Cloud [paper](https://arxiv.org/pdf/2003.01251.pdf), [code](https://github.com/WeijingShi/Point-GNN)  
* 2021 S3Net: 3D LiDAR Sparse Semantic Segmentation Network [paper](https://arxiv.org/pdf/2103.08745.pdf).  
* 2021 LSNet: Learned Sampling Network for 3D Object Detection from Point Clouds [paper](https://www.mdpi.com/2072-4292/14/7/1539).  
* 2021 Pyramid R-CNN:Towards Better Performance and Adaptability for 3D Object Detection [paper](https://arxiv.org/pdf/2109.02499.pdf), [code](https://github.com/PointsCoder/Pyramid-RCNN)  
* 2021 ST3D: Self-training for Unsupervised Domain Adaptation on 3D Object Detection [paper](https://arxiv.org/pdf/2103.05346.pdf), [code](https://github.com/CVMI-Lab/ST3D)  
* 2021 POAT-Net: Parallel Offset-Attention Assisted Transformer for 3D Object Detection for Autonomous Driving. [paper](https://ieeexplore.ieee.org/iel7/6287639/9312710/09611257.pdf). 
* 2022 SASA: Semantics-Augmented Set Abstraction for Point-based 3D Object Detection [paper](https://arxiv.org/pdf/2201.01976.pdf), [code](https://github.com/blakechen97/SASA)  
* 2022 PointDistiller: Structured Knowledge Distillation Towards Efficient and Compact 3D Detection [paper](https://arxiv.org/pdf/2205.11098.pdf), [code](https://github.com/RunpeiDong/PointDistiller)  

**Projection-based**  
* 2015 Multi-view Convolutional Neural Networks for 3D Shape Recognition [paper](https://arxiv.org/pdf/1505.00880.pdf), [code](https://github.com/suhangpro/mvcnn).    
* 2017 DepthCN: Vehicle detection using 3D-LIDAR and ConvNet [paper](https://ieeexplore.ieee.org/document/8317880/).     
* 2017 MV3D: Multi-View 3D Object Detection Network for Autonomous Driving[paper](https://arxiv.org/pdf/1611.07759.pdf), [code](https://github.com/bostondiditeam/MV3D).     
* 2018 GVCNN: Group-View Convolutional Neural Networks for 3D Shape Recognition [paper](https://openaccess.thecvf.com/content_cvpr_2018/papers/Feng_GVCNN_Group-View_Convolutional_CVPR_2018_paper.pdf), [code](https://github.com/waxnkw/gvcnn-pytorch).    
* 2018 PointSeg: Real-Time Semantic Segmentation Based on 3D LiDAR Point Cloud [paper](https://arxiv.org/pdf/1807.06288.pdf), [code](https://github.com/ArashJavan/PointSeg)  
* 2018 SnapNet: 3D point cloud semantic labeling with 2D deep segmentation networks [paper](https://boulch.eu/files/2017_cag_snapNet.pdf), [code](https://github.com/aboulch/snapnet)  
* 2018 SqueezeSeg:  Convolutional Neural Nets with Recurrent CRF for Real-Time Road-Object Segmentation from 3D LiDAR Point Cloud [paper](https://arxiv.org/pdf/1710.07368.pdf), [code](https://github.com/BichenWuUCB/SqueezeSeg)  
* 2018 RT3D: Real-Time 3-D Vehicle Detection in LiDAR Point Cloud for Autonomous Driving [paper](https://ieeexplore.ieee.org/iel7/7083369/7339444/08403277.pdf), [code](https://github.com/zyms5244/RT3D)   
* 2018 BirdNet: a 3D Object Detection Framework from LiDAR information [paper](https://arxiv.org/pdf/1805.01195.pdf)   
* 2018 PIXOR: Real-time 3D Object Detection from Point Clouds [paper](https://arxiv.org/pdf/1902.06326.pdf), [code](https://github.com/matssteinweg/PIXOR)   
* 2018 Complex-YOLO: An Euler-Region-Proposal for Real-time 3D Object Detection on Point Clouds [paper](https://arxiv.org/pdf/1803.06199.pdf), [code](https://github.com/AI-liu/Complex-YOLO)   
* 2019 SqueezeSeg v2: Improved Model Structure and Unsupervised Domain Adaptation for Road-Object Segmentation from a LiDAR Point Cloud [paper](https://arxiv.org/pdf/1809.08495.pdf), [code](https://github.com/xuanyuzhou98/SqueezeSegV2)  
* 2020 BirdNet+: End-to-End 3D Object Detection in LiDAR Bird’s Eye View [paper](https://arxiv.org/pdf/2003.04188.pdf), [code](https://github.com/AlejandroBarrera/birdnet2)    
* 2020 MVLidarNet: Real-Time Multi-Class Scene Understanding for Autonomous Driving Using Multiple Views [paper](https://arxiv.org/pdf/2006.05518.pdf).     
* 2020 RAANet: Range-Aware Attention Network for LiDAR-based 3D Object Detection with Auxiliary Point Density Level Estimation [paper](https://arxiv.org/pdf/2111.09515.pdf), [code](https://github.com/anonymous0522/raan)  
* 2022 Pseudo-L: Pseudo-Stereo for Monocular 3D Object Detection in Autonomous Driving [paper](https://arxiv.org/pdf/2203.02112.pdf), [code](https://github.com/revisitq/Pseudo-Stereo-3D)   
* 2020 SRDL: Stereo RGB and Deeper LIDAR Based Network for 3D Object Detection [paper](https://arxiv.org/pdf/2006.05187.pdf).   
* 2023 RI-Fusion: 3D Object Detection Using Enhanced Point Features With Range-Image Fusion for Autonomous Driving [paper](https://ieeexplore.ieee.org/document/9963928).  

**Voxel-based**  
* 2015 VoxNet: A 3D Convolutional Neural Network for real-time object recognition [paper](https://ieeexplore.ieee.org/document/7353481), [code](https://github.com/AutoDeep/VoxNet)  
* 2017 3DFCN: High-Resolution Shape Completion Using Deep Neural Networks for Global Structure and Local Geometry Inference [paper](https://arxiv.org/pdf/1709.07599.pdf)  
* 2017 SEGCloud: Semantic Segmentation of 3D Point Clouds [paper](https://arxiv.org/pdf/1710.07563.pdf)   
* 2017 OctNet: Learning Deep 3D Representations at High Resolutions [paper](https://arxiv.org/pdf/1611.05009.pdf), [code](https://github.com/griegler/octnet)  
* 2017 Kd-Net: Escape from Cells: Deep Kd-Networks for the Recognition of 3D Point Cloud Models [paper](https://arxiv.org/pdf/1704.01222.pdf), [code](https://github.com/fxia22/kdnet.pytorch)  
* 2018 VoxelNet: End-to-End Learning for Point Cloud Based 3D Object Detection [paper](https://arxiv.org/pdf/1711.06396.pdf), [code](https://github.com/steph1793/Voxelnet)  
* 2018 3DmFV: 3D Point Cloud Classification and Segmentation using 3D Modified Fisher Vector Representation for Convolutional Neural Networks [paper](https://arxiv.org/pdf/1711.08241.pdf), [code](https://github.com/sitzikbs/3DmFV-Net)  
* 2018 PointGrid: A Deep Network for 3D Shape Understanding [paper](https://openaccess.thecvf.com/content_cvpr_2018/papers/Le_PointGrid_A_Deep_CVPR_2018_paper.pdf), [code](https://github.com/trucleduc/PointGrid)  
* 201MVX-Net: Multimodal VoxelNet for 3D Object Detection [paper](https://arxiv.org/pdf/1904.01649.pdf), [code](https://github.com/AIR-THU/DAIR-V2X/blob/main/configs/sv3d-veh/mvxnet/README.md)   
* 2019 F-pointNet: Frustum PointNets for 3D Object Detection from RGB-D Data [paper](https://arxiv.org/pdf/1711.08488.pdf), [code](https://github.com/charlesq34/frustum-pointnets)  
* 2020 PV-RCNN: Point-Voxel Feature Set Abstraction for 3D Object Detection [paper](https://arxiv.org/pdf/1912.13192.pdf), [code](https://github.com/sshaoshuai/PV-RCNN)   
* 2020 SECOND: Sparsely Embedded Convolutional Detection [paper](https://www.mdpi.com/1424-8220/18/10/3337), [code](https://github.com/open-mmlab/OpenPCDet)  
* 2020 HVNet: Hybrid Voxel Network for LiDAR Based 3D Object Detection [paper](https://arxiv.org/pdf/2003.00186.pdf), [code](https://github.com/AndyYuan96/HVNet)  
* 2020 Associate-3Ddet [paper](https://arxiv.org/pdf/2006.04356.pdf)   
* 2021 Voxel R-CNN: Towards High Performance Voxel-based 3D Object Detection [paper](https://arxiv.org/pdf/2012.15712.pdf), [code](https://github.com/djiajunustc/Voxel-R-CNN)   
* 2021 TANet: Robust 3D Object Detection from Point Clouds with Triple Attention [paper](https://arxiv.org/pdf/1912.05163.pdf), [code](https://github.com/happinesslz/TANet)   
* 2021 VIN: Voxel-based Implicit Network for Joint3D Object Detection and Segmentation for Lidars [paper](https://arxiv.org/pdf/2107.02980.pdf)  
* 2022 SIENet: Spatial Information Enhancement Network for 3D Object Detection from Point Cloud [paper](https://arxiv.org/pdf/2103.15396.pdf), [code](https://github.com/Liz66666/SIENet)   
* 2022 SMS-Net: Sparse multi-scale voxel feature aggregation network for LiDAR-based 3D object detection [paper](https://www.sciencedirect.com/science/article/abs/pii/S092523122200786X)   
* 2022 MA-MFFC:3D Object Detection Based on Attention and Multi-Scale Feature Fusion [paper](https://pubmed.ncbi.nlm.nih.gov/35632344/)   
* 2022 PDV: Point Density-Aware Voxels for LiDAR 3D Object Detection [paper](https://arxiv.org/pdf/2203.05662.pdf), [code](https://github.com/TRAILab/PDV)   
* 2023 SAT-GCN: Self-attention graph convolutional network-based 3D object detection for autonomous driving [paper](https://www.sciencedirect.com/science/article/pii/S0950705122011765)    

**Pillar-based**  
* 2019 PointPillars: Fast Encoders for Object Detection from Point Clouds [paper](https://arxiv.org/pdf/1812.05784.pdf), [code](https://github.com/zhulf0804/PointPillars)   
* 2022 Associate-3Ddet: Perceptual-to-Conceptual Association for 3D Point Cloud Object Detection [paper](https://www.sciencedirect.com/science/article/abs/pii/S0925231221019184)    
* 2022 WCNN3D: Wavelet Convolutional Neural Network-Based 3D Object Detection for Autonomous Driving [paper](https://www.mdpi.com/1424-8220/22/18/7010)   

**Hybrid-based**  
* 2020 SegVoxelNet: Exploring Semantic Context and Depth-aware Features for 3D Vehicle Detection from Point Cloud [paper](https://arxiv.org/pdf/2002.05316.pdf)     
* 2020 AMVNet: Assertion-based Multi-View Fusion Network for LiDAR Semantic Segmentation [paper](https://arxiv.org/pdf/2012.04934.pdf)     
* 2020 FusionNet: A deep fully residual convolutional neural network for image segmentation in connectomics [paper](https://arxiv.org/ftp/arxiv/papers/1612/1612.05360.pdf)     
* 2021 AF2-S3Net: Attentive Feature Fusion with Adaptive Feature Selection for Sparse Semantic Segmentation Network [paper](https://openaccess.thecvf.com/content/CVPR2021/papers/Cheng_AF2-S3Net_Attentive_Feature_Fusion_With_Adaptive_Feature_Selection_for_Sparse_CVPR_2021_paper.pdf)     
* 2021 Sparse Single Sweep LiDAR Point Cloud Segmentation via Learning Contextual Shape Priors from Scene Completion [paper](https://arxiv.org/pdf/2012.03762.pdf), [code](https://github.com/yanx27/JS3C-Net)    

## 3. RADAR deep-learning-based method 
Facilitated by the antenna array processing, the mmWave radar can obtain the angular information of vehicle reflecting points, 
which, combined with the time of flight (ToF), can localize the points in 3D space. Compared with other sensors such as LiDAR and
cameras, mmWave radar signals are less affected by severe weather conditions (except heavy rain) and are insensitive to lighting.
Depending on the signal output, mmWave-based vehicle detection methods can be classified into two categories. These are raw-data-level 
and map-level radar detection.

**1. Raw data level radar detection**
MmWave raw data contain the required information to
locate the detected vehicles. Nowadays, with the development of radar technology, most radars, after receiving
the reflected signal can directly determine the detected target information such as range, velocity, angle, and
reflection intensity of detected vehicles.

**2. Map-level radar target detection**

Compared to raw data target detection radar, the imaging radar not only provides the target’s velocity and
motion state but also outputs the radar signals into image maps.


**Below are some vehicle detection related works**  

* Point cloud features-based kernel svm for human\-vehicle classification in millimeter wave radar, [paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=8979449)
* Radarnet: Exploiting radar for robust perception of dynamic objects, [paper](https://arxiv.org/pdf/2007.14366.pdf?utm_channel=SOCIAL)
* Through fog high-resolution imaging using millimeter wave radar, [paper](http://openaccess.thecvf.com/content_CVPR_2020/papers/Guan_Through_Fog_High-Resolution_Imaging_Using_Millimeter_Wave_Radar_CVPR_2020_paper.pdf)
* Exploiting temporal relations on radar perception for autonomous driving, [paper](https://openaccess.thecvf.com/content/CVPR2022/papers/Li_Exploiting_Temporal_Relations_on_Radar_Perception_for_Autonomous_Driving_CVPR_2022_paper.pdf)
* 3D point cloud generation with millimeter-wave radar [paper](https://dl.acm.org/doi/pdf/10.1145/3432221)
* Radar-only ego-motion estimation in difficult settings via graph matching [paper](https://arxiv.org/pdf/1904.11476.pdf)
* Automotive radar the key technology for autonomous driving: From detection and ranging to environmental understanding [paper](https://www.researchgate.net/profile/Juergen-Dickmann/publication/301960180_Automotive_Radar_the_Key_Technology_For_Autonomous_Driving_From_Detection_and_Ranging_to_Environmental_Understanding/links/5742cff908aea45ee84a6ede/Automotive-Radar-the-Key-Technology-For-Autonomous-Driving-From-Detection-and-Ranging-to-Environmental-Understanding.pdf)
* Deep instance segmentation with automotive radar detection points [paper](https://arxiv.org/pdf/2110.01775)
* Iterative adaptive approaches to MIMO radar imaging [paper](http://www.sal.ufl.edu/papers/MIMOIAA.pdf)
* Mmwave radar point cloud segmentation using gmm in multimodal traffic monitoring [paper](https://arxiv.org/pdf/1911.06364)
* RSS-Net: weakly-supervised multi-class semantic segmentation with FMCW radar [paper](https://arxiv.org/pdf/2004.03451)
* Point cloud features-based kernel SVM for human vehicle classification in millimeter wave radar [pdf](https://ieeexplore.ieee.org/iel7/6287639/6514899/08979449.pdf)


## 4. Multi-sensor fusion-based method for vehicle detection  

The information from a single sensor cannot guarantee the safety requirement and the degree of accuracy required for an 
intelligent vehicle in complex autonomous environment perception.  Every sensor has its pros and cons in terms of 
environment perception. these three sensors have complementary advantages. Therefore, multi-sensor fusion is an 
inevitable topic in autonomous driving.

### Stereo vision-based vehicle detection methods
In the stereo vision system, vehicle detection is performed on information obtained from a combination of two or more 
cameras capturing different perspectives of the given scene.

**Below are some vehicle detection related works** 
* Vehicle detection by means of stereo vision-based obstacles features extraction and monocular pattern analysis [paper](https://www.ce.unipr.it/people/broggi/publications/ieee.tip.2006.pdf)
* Vehicle detection system design based on stereo vision sensors [paper](https://link.springer.com/article/10.1007/s12239-009-0043-z)
* Vehicle detection and tracking using mean shift segmentation on semi-dense disparity maps [paper](https://ieeexplore.ieee.org/abstract/document/6232280/)
* Online vehicle detection using Haar-like, LBP and HOG feature based image classifiers with stereo vision preselection [paper](http://bib.drgoehring.de/neumann17iv-vehicledetection/neumann17iv-vehicledetection.pdf)
* Real-time obstacle detection in complex scenarios using dense stereo vision and optical flow [paper](https://ieeexplore.ieee.org/abstract/document/5625174)
* 3D pose estimation of vehicles using stereo camera [paper](https://citeseerx.ist.psu.edu/document?repid=rep1&type=pdf&doi=c5589374a8056b5fef3af8bef7858677fc273ac6)

### Vehicle detection based on the fusion of mmWave radar and vision
The mmWave radar and the cameras are the classical vehicular sensors commonly used for vehicle detection andtracking. 
The mmWave radar is known to provide depth information and the motion state of the vehicle. In contrast, the camera image 
is known to have a high resolution containing rich semantic information. Despite their promising abilities, their drawbacks 
are insufficient for environmental perception when used separately. Therefore, fusing mmWave radar and the camera compliments 
each other for better environment perception of the ego vehicle.

![Fig 8](https://github.com/JulesK8/Vehicle-Detection/assets/134627326/6b7a2259-de63-4038-b914-749911567405)


**Data level fusion**
* Pedestrian detection based on fusion of millimeter wave radar and vision [paper](https://dl.acm.org/doi/abs/10.1145/3268866.3268868)
* On-road vehicle detection and tracking using MMW radar and monovision fusion [paper](https://ieeexplore.ieee.org/abstract/document/7463071/)
* CramNet: Camera-Radar Fusion with Ray Constrained Cross-Attention for Robust 3D Object Detection [paper](https://arxiv.org/pdf/2210.09267)
* 
**Decision level fusion**
* Standard platform for sensor fusion on advanced driver assistance system using bayesian network [paper](https://ieeexplore.ieee.org/abstract/document/1336390/)
* Camera radar fusion for increased reliability in ADAS applications [paper](https://library.imaging.org/admin/apis/public/api/ist/website/downloadArticle/ei/30/17/art00011)
* Frontal object perception using radar and mono-vision [paper](https://citeseerx.ist.psu.edu/document?repid=rep1&type=pdf&doi=a580214d5e97f982305dcc6e00d2441ea3438dbe)
* Appearance based vehicle detection by radar-stereo vision integration [paper](https://link.springer.com/chapter/10.1007/978-3-319-27146-0_34)

**Feature level**
* Deep fusionnet for point cloud semantic segmentation [paper](https://ora.ox.ac.uk/objects/uuid:80c17ed9-01ef-486e-bea5-962cc4b56528/download_file?safe_filename=Deep%2520FusionNet.pdf&type_of_work=Conference+item)
* Distant vehicle detection using radar and vision [paper](https://arxiv.org/pdf/1901.10951)
* RVNet: Deep sensor fusion of monocular camera and radar for image-based obstacle detection in challenging environments [paper](https://www.researchgate.net/profile/Vijay-John/publication/335833918_RVNet_Deep_Sensor_Fusion_of_Monocular_Camera_and_Radar_for_Image-based_Obstacle_Detection_in_Challenging_Environments/links/5d7f164e92851c87c38b09f1/RVNet-Deep-Sensor-Fusion-of-Monocular-Camera-and-Radar-for-Image-based-Obstacle-Detection-in-Challenging-Environments.pdf)
* A deep learning-based radar and camera sensor fusion architecture for object detection [paper](https://arxiv.org/pdf/2005.07431)
* Spatial attention fusion for obstacle detection using mmwave radar and vision sensor [paper](https://www.mdpi.com/1424-8220/20/4/956/pdf)
* Automotive radar and camera fusion using generative adversarial networks [paper](https://www.sciencedirect.com/science/article/pii/S1077314219300530)
* Radar and camera early fusion for vehicle detection in advanced driver assistance systems [paper](https://ml4ad.github.io/files/papers/Radar%20and%20Camera%20Early%20Fusion%20for%20Vehicle%20Detection%20in%20Advanced%20Driver%20Assistance%20Systems.pdf)
* Bridging the view disparity between radar and camera features for multi-modal fusion 3d object detection [paper](https://arxiv.org/pdf/2208.12079.pdf)

## LiDAR and vision fusion-based vehicle detection
LiDAR vision-based vehicle detection methodologies can be classified into two classes based on different fusion stages.
These are early fusion and late fusion. Early fusion can be categorized as data-level fusion and feature-level fusion,
whereas late fusion can be classified as decision-level fusion and multi-level fusion.

![Fig 9](https://github.com/JulesK8/Vehicle-Detection/assets/134627326/4b4cabfe-bffa-4cd5-b4dc-a017ce5c0458)



**Some of representative works**
* Real-time vehicle detection framework based on the fusion of LiDAR and camera [paper](https://www.mdpi.com/2079-9292/9/3/451)
* Dual-view 3D object recognition and detection via Lidar point cloud and camera image [paper](https://www.sciencedirect.com/science/article/pii/S0921889021002542)
* Deep structural information fusion for 3D object detection on LiDAR–camera system [paper](https://www.sciencedirect.com/science/article/abs/pii/S1077314221001399)
* Real-time vehicle detection algorithm based on vision and lidar point cloud fusion [paper](https://www.hindawi.com/journals/js/2019/8473980/abs/)
* Fast and accurate 3D object detection for lidar-camera-based autonomous vehicles using one shared voxel-based backbone [paper](https://ieeexplore.ieee.org/iel7/6287639/6514899/09340187.pdf)
* Fusion of 3D LIDAR and camera data for object detection in autonomous vehicle applications [pdf](https://scholarworks.utrgv.edu/cgi/viewcontent.cgi?article=1021&context=cs_fac)
* Target fusion detection of LiDAR and camera based on the improved YOLO algorithm [paper](https://www.mdpi.com/2227-7390/6/10/213/pdf)
* 3D vehicle detection using multi-level fusion from point clouds and images [paper](https://www.researchgate.net/profile/Lingfei-Ma/publication/357753602_3D_Vehicle_Detection_Using_Multi-Level_Fusion_From_Point_Clouds_and_Images/links/61de11ce323a2268f99ad4f8/3D-Vehicle-Detection-Using-Multi-Level-Fusion-From-Point-Clouds-and-Images.pdf)
* DeepFusionMOT: A 3D Multi-Object Tracking Framework Based on Camera-LiDAR Fusion With Deep Association [paper](https://arxiv.org/pdf/2202.12100)
* Tightly-coupled LIDAR and computer  vision integration for vehicle detection [paper](https://www.researchgate.net/profile/Matthew-Barth/publication/224562486_Tightly-Coupled_LIDAR_and_Computer_Vision_Integration_for_Vehicle_Detection/links/55bb9ff208aec0e5f4418f47/Tightly-Coupled-LIDAR-and-Computer-Vision-Integration-for-Vehicle-Detection.pdf)
* Multi-view 3d object detection network for autonomous driving [paper](https://openaccess.thecvf.com/content_cvpr_2017/papers/Chen_Multi-View_3D_Object_CVPR_2017_paper.pdf)
