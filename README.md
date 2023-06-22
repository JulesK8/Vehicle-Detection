# Vehicle-Detection
AWESOME VEHICLE DETECTION ALGORTHMS AND DATASETS


A curated list of existing methods and datasets for vehicle detection.  
You are welcome to update the list.  
Author: Jules KARANGWA  
Contact: karangwa@mail.ustc.edu.cn  

> **Note:**   
> A review paper related to this list is expected to be published in IEEE Transactions.  
The paper is currently under review. I will keep updating this repo for the latest works in  
vehicle detection field, and i will provide the link for the paper after publication.   
If you find that the paper contents are useful, please cite this paper in your work.  

# Intro  
Efficient and accurate vehicle detection is one of the essential tasks in the environment  
perception of an autonomous vehicle. Therefore, numerous  algorithms for vehicle detection  
have been developed. 

# Datasets  
A more detailed table about datasets used in vehicle detection can be found on page 4 and 5 of my  
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

***Appearance-based methods***  

**a. Symmetry-based methods**  
* [Multipart vehicle detection using symmetry-derived analysis and active learning](https://ieeexplore.ieee.org/abstract/document/7368153/)
* [Symmetry-based monocular vehicle detection system](https://link.springer.com/article/10.1007/s00138-011-0355-7) 
* [Video vehicle detection algorithm based on edge symmetry](https://bhxb.buaa.edu.cn/bhzk/en/article/id/9405)  

**b. Edge feature-based methods**
* [Edge-based forward vehicle detection method for complex scenes](https://ieeexplore.ieee.org/abstract/document/6904044/)  
* [Multiscale edge fusion for vehicle detection based on difference of Gaussian](https://www.sciencedirect.com/science/article/pii/S0030402616000292)  
* [Vehicle  detection based on underneath vehicle shadow using edge features](https://ieeexplore.ieee.org/abstract/document/7893608/)  
* [A feature-based recognition scheme for traffic scenes](https://ieeexplore.ieee.org/abstract/document/528285/)  

**c. Color-based methods**
* [Color modeling by spherical influence field in sensing driving environment](https://ieeexplore.ieee.org/abstract/document/898350/)  
* [Daytime preceding vehicle brake light detection using monocular vision](https://ieeexplore.ieee.org/abstract/document/7247631)  
* [Vehicle detection based on color analysis](https://ieeexplore.ieee.org/abstract/document/6380975/)  
* [Road detection and vehicles tracking by vision for an on-board acc system in the velac vehicle](https://ieeexplore.ieee.org/abstract/document/859833)

**d. Taillight-based methods**
* [Rear-lamp vehicle detection and tracking in low-exposure color video for night conditions](https://ieeexplore.ieee.org/abstract/document/5446402/)  
* [Vision-based nighttime vehicle detection using CenSurE and SVM](https://ieeexplore.ieee.org/abstract/document/7103307)  
* [An improved technique for Night-time Vehicle detection](https://ieeexplore.ieee.org/abstract/document/8554712)  
* [Looking at vehicles in the night: Detection and dynamics of rear lights](https://ieeexplore.ieee.org/abstract/document/7750549)  
* [Vehicle detection using tail light segmentation](https://ieeexplore.ieee.org/abstract/document/6021126)  

**e. Underneath shadow-based methods**
* [Vehicle detection based on underneath vehicle shadow using edge features](https://ieeexplore.ieee.org/abstract/document/7893608/)  
* [Shadow-based vehicle detection in urban traffic](https://www.mdpi.com/193936)  
* [Shadow Based On-Road Vehicle Detection and Verification Using HAAR Wavelet Packet Transform](https://ieeexplore.ieee.org/abstract/document/1598621/)  
* [Shadow detection in camera-based vehicle detection: survey and analysis](https://www.spiedigitallibrary.org/journals/Journal-of-Electronic-Imaging/volume-25/issue-5/051205/Shadow-detection-in-camera-based-vehicle-detection--survey-and/10.1117/1.JEI.25.5.051205.short?SSO=1)  

**f. Texture-based methods**
* [Robust vehicle detection in vision systems based on fast wavelet transform and texture analysis](https://ieeexplore.ieee.org/abstract/document/4339088/)  
* [Real-time small obstacle detection on highways using compressive RBM road reconstruction](https://ieeexplore.ieee.org/abstract/document/7225680/) 

***Stereo-vision-based methods***    
* [GOLD: A parallel real-time stereo vision system for generic obstacle and lane detection](https://ieeexplore.ieee.org/abstract/document/650851/)  
* [Vehicle detection based on multifeature extraction and recognition adopting RBF neural network on ADAS system](https://www.hindawi.com/journals/complexity/2020/8842297/)  

***Motion-based methods***    
* [Motion based vehicle detection on motorways](https://ieeexplore.ieee.org/abstract/document/528329/)  
* [Motion-based vehicle detection in Hsuehshan Tunnel](https://ieeexplore.ieee.org/abstract/document/7449856/) 
* [Moving vehicle detection and tracking in unstructured environments](https://ieeexplore.ieee.org/abstract/document/6224636/)  
* [Go with the flow: Improving Multi-View vehicle detection with motion cues](https://ieeexplore.ieee.org/abstract/document/6977422/)  

**2. Machine learning-based methods**  

**a. HOG:**   
* [Histograms of oriented gradients](https://courses.cs.duke.edu/fall17/compsci527/notes/hog.pdf)  
* [Real-time vehicle detection using histograms of oriented gradients and AdaBoost classification](https://www.sciencedirect.com/science/article/pii/S0030402616305423)  
* [Front and rear vehicle detection using hypothesis generation and verification](https://www.academia.edu/download/37934652/4413sipij03.pdf)  
* [Image-based on-road vehicle detection using cost-effective histograms of oriented gradients](https://www.sciencedirect.com/science/article/abs/pii/S1047320313001478)  
* [Vision-based vehicle detection system with consideration of the detecting location](https://ieeexplore.ieee.org/abstract/document/6175131/)  

**b. Haar-like feature:**  
* [Rapid object detection using a boosted cascade of simple features](https://ieeexplore.ieee.org/abstract/document/990517/)  
* [Efficient feature selection and classification for vehicle detection](https://ieeexplore.ieee.org/abstract/document/6898836/)  
* [A general active-learning framework for on-road vehicle recognition and tracking](https://ieeexplore.ieee.org/abstract/document/5411825)  
* [A vehicle detection system based on haar and triangle features](https://ieeexplore.ieee.org/abstract/document/5164288/)  
* [Monocular precrash vehicle detection: features and classifiers](https://ieeexplore.ieee.org/abstract/document/1643708/)  
* [SURF: Speed Up Robust Features](https://lirias.kuleuven.be/73068?limo=0)  
* [Integrating appearance and edge features for sedan vehicle detection in the blind-spot area](https://ieeexplore.ieee.org/abstract/document/6145682/)  
* [Object recognition from local scale-invariant feature](https://ieeexplore.ieee.org/abstract/document/790410/)    
* [Vehicle detection using an extended hidden random field mode](https://ieeexplore.ieee.org/abstract/document/6083135/)  

**3. Deep-learning-based methods for vehicle detection**  

**_A. Objectdetection-based methods_**  

**1. Anchor-based detectors**  
**a. Two-stage detection networks**  
* [RCNN](http://openaccess.thecvf.com/content_cvpr_2014/html/Girshick_Rich_Feature_Hierarchies_2014_CVPR_paper.html)  
* [Fast RCNN](http://openaccess.thecvf.com/content_iccv_2015/html/Girshick_Fast_R-CNN_ICCV_2015_paper.html)  
* [Faster RCNN](https://proceedings.neurips.cc/paper/2015/hash/14bfa6bb14875e45bba028a21ed38046-Abstract.html)  
* [SPP-Net](https://ieeexplore.ieee.org/abstract/document/7005506/)  

> Note:  
> These are some pioneering networks in context of object detection, i.e, these networks are not only used to detect  
> vehicles but also detect other objects. Consider taking a look into my review paper for a deep understanding.

**b. One-stage detection networks**  
* [YOLO v1](https://www.cv-foundation.org/openaccess/content_cvpr_2016/html/Redmon_You_Only_Look_CVPR_2016_paper.html)   
* [SSD](https://link.springer.com/chapter/10.1007/978-3-319-46448-0_2) 
* [YOLO v2](http://openaccess.thecvf.com/content_cvpr_2017/html/Redmon_YOLO9000_Better_Faster_CVPR_2017_paper.html)    
* [YOLO v3](https://arxiv.org/abs/1804.02767)    
* [M2Det](https://ojs.aaai.org/index.php/AAAI/article/view/4962)  
* [YOLO v4](https://arxiv.org/abs/2004.10934)    
* [YOLO v7](http://openaccess.thecvf.com/content/CVPR2023/html/Wang_YOLOv7_Trainable_Bag-of-Freebies_Sets_New_State-of-the-Art_for_Real-Time_Object_Detectors_CVPR_2023_paper.html)  

> Note:  
> These are some pioneering networks in context of object detection, i.e, these networks are not only used to detect  
> vehicles but also detect other objects. Consider taking a look into my review paper for a deep understanding.  

**2. Anchor-free detectors**  
The existing studies show that anchor-free models can be classified as key-point- and center-based. Key-point-based  
methods detect specific object points, such as center and corner points, and group them for bounding box prediction.   
whereas, center-based methods directly predict the object's center point and perform the object bounding box regression.  

**a. Anchor-free keypoint-based**  
* [ExtremeNet](http://openaccess.thecvf.com/content_CVPR_2019/html/Zhou_Bottom-Up_Object_Detection_by_Grouping_Extreme_and_Center_Points_CVPR_2019_paper.html)  
* [CornerNet](http://openaccess.thecvf.com/content_ECCV_2018/html/Hei_Law_CornerNet_Detecting_Objects_ECCV_2018_paper.html)  
* [CenterNet-HG](https://arxiv.org/abs/1904.07850)  
* [Grid RCNN](http://openaccess.thecvf.com/content_CVPR_2019/html/Lu_Grid_R-CNN_CVPR_2019_paper.html)
* [CornerNet-Lite](https://arxiv.org/abs/1904.08900)
* [CenterNet](http://openaccess.thecvf.com/content_ICCV_2019/html/Duan_CenterNet_Keypoint_Triplets_for_Object_Detection_ICCV_2019_paper.html)
* [RepPoints](http://openaccess.thecvf.com/content_ICCV_2019/html/Yang_RepPoints_Point_Set_Representation_for_Object_Detection_ICCV_2019_paper.html)

**b. Anchor-free center-based**  
* [FCOS](https://openaccess.thecvf.com/content_ICCV_2019/html/Tian_FCOS_Fully_Convolutional_One-Stage_Object_Detection_ICCV_2019_paper.html)  
* [Foveabox](https://ieeexplore.ieee.org/abstract/document/9123553/)
* [FSAF](http://openaccess.thecvf.com/content_CVPR_2019/html/Zhu_Feature_Selective_Anchor-Free_Module_for_Single-Shot_Object_Detection_CVPR_2019_paper.html)
* [GA-RPN](http://openaccess.thecvf.com/content_CVPR_2019/html/Wang_Region_Proposal_by_Guided_Anchoring_CVPR_2019_paper.html)
* [ATSS](https://openaccess.thecvf.com/content_CVPR_2020/html/Zhang_Bridging_the_Gap_Between_Anchor-Based_and_Anchor-Free_Detection_via_Adaptive_CVPR_2020_paper.html)  

**c. Transformer-based detector**
* [DGT-Det3D](https://www.sciencedirect.com/science/article/abs/pii/S0950705122011819)  
* [ET-DTH](https://arxiv.org/abs/2303.05078)  
* [RDSA](https://www.hindawi.com/journals/cin/2023/4228610/)  
* [VoxSet](https://openaccess.thecvf.com/content/CVPR2022/html/He_Voxel_Set_Transformer_A_Set-to-Set_Approach_to_3D_Object_Detection_CVPR_2022_paper.html)  
* [VISTA](http://openaccess.thecvf.com/content/CVPR2022/html/Deng_VISTA_Boosting_3D_Object_Detection_via_Dual_Cross-VIew_SpaTial_Attention_CVPR_2022_paper.html)  
* [CenterFormer](https://link.springer.com/chapter/10.1007/978-3-031-19839-7_29)  
* [SW Former](https://link.springer.com/chapter/10.1007/978-3-031-20080-9_25)  
* [STVD](https://www.sciencedirect.com/science/article/pii/S0957417422020103)  
* [VoTr](http://openaccess.thecvf.com/content/ICCV2021/html/Mao_Voxel_Transformer_for_3D_Object_Detection_ICCV_2021_paper.html)  
* [MFT-SST](https://ieeexplore.ieee.org/abstract/document/10042064/)  

> Note:  
> The above models are inspired by the original work presented as [Vit](https://arxiv.org/abs/2010.11929).  

**_B. Semantic segmentation-based detection networks_**  
Semantic segmentation associates a label or a category with every pixel in the image.  

>Note:  
>It can be mentioned that most of semantic segmentation methods are applicable in the field of vehicle detection.

* [DeepMask](https://www.cv-foundation.org/openaccess/content_cvpr_2016/html/Szegedy_Rethinking_the_Inception_CVPR_2016_paper.html)  
* [DeepLab](https://ieeexplore.ieee.org/abstract/document/7913730/)  
* [DeepLab2](https://arxiv.org/abs/2106.09748)  
* [DeepLab3](https://arxiv.org/abs/1706.05587)  
* [PSPNet](http://openaccess.thecvf.com/content_cvpr_2017/html/Zhao_Pyramid_Scene_Parsing_CVPR_2017_paper.html)  
* [ICNet](http://openaccess.thecvf.com/content_ECCV_2018/html/Hengshuang_Zhao_ICNet_for_Real-Time_ECCV_2018_paper.html)  
* [RefineNet](http://openaccess.thecvf.com/content_cvpr_2017/html/Lin_RefineNet_Multi-Path_Refinement_CVPR_2017_paper.html)  
* [Semi Supervised Semantic Segmentation Using GAN](https://openaccess.thecvf.com/content_iccv_2017/html/Souly__Semi_Supervised_ICCV_2017_paper.html)  
* [SegNet](https://ieeexplore.ieee.org/abstract/document/7803544/)  
* [BiseNet](http://openaccess.thecvf.com/content_ECCV_2018/html/Changqian_Yu_BiSeNet_Bilateral_Segmentation_ECCV_2018_paper.html)  
* [DFANet](http://openaccess.thecvf.com/content_CVPR_2019/html/Li_DFANet_Deep_Feature_Aggregation_for_Real-Time_Semantic_Segmentation_CVPR_2019_paper.html)  
* [Enet](https://arxiv.org/abs/1606.02147)  
* [LedNet](https://ieeexplore.ieee.org/abstract/document/8803154/)  
* [EspNet](https://arxiv.org/abs/1804.00015)  
* [CGNet](https://ieeexplore.ieee.org/abstract/document/9292449)  

## **LiDAR-based methods for vehicle detection**  

LiDAR is a sensor that perceives the environment by transmitting laser signals and receiving the reflected signals   
from the obstacles in the environment.  

Practically, existing LiDAR-based methods can be classified as traditional and deep learning techniques.  

**1. Traditional methods**  

* [A real-time grid map generation and object classification for ground-based 3D LIDAR data using image analysis techniques](https://ieeexplore.ieee.org/abstract/document/5651197/) 
* [A Graph-Based Method for Joint Instance Segmentation of Point Clouds and Image Sequences](https://ieeexplore.ieee.org/abstract/document/9560765)  
* [Range image-based density-based spatial clustering of application with noise clustering method of three-dimensional point clouds](https://journals.sagepub.com/doi/pdf/10.1177/1729881418762302)  
* [Improving LiDAR point cloud classification using intensities and multiple echoes](https://ieeexplore.ieee.org/abstract/document/7354098)  
* [Drivable road detection with 3D point clouds based on the MRF for intelligent vehicle,](https://link.springer.com/chapter/10.1007/978-3-319-07488-7_4)  

**2. LiDAR deep-learning-based method**  
Considering the data type where we extract featuremaps, LiDAR deep learning methods can be categorized as  
raw data-based, projection-based, and voxel-based.

**Raw data based**
* 2017 PointNet [paper](https://openaccess.thecvf.com/content_cvpr_2017/papers/Qi_PointNet_Deep_Learning_CVPR_2017_paper.pdf), [code](https://github.com/charlesq34/pointnet)  
* 2017 PointNet++ [paper](https://proceedings.neurips.cc/paper/2017/file/d8bf84be3800d12f74d8b05e9b89836f-Paper.pdf), [code](https://github.com/charlesq34/pointnet2)  
* 2018 IPOD [paper](https://arxiv.org/pdf/1812.05276.pdf)   
* 2018 RoarNet [paper](https://arxiv.org/pdf/1811.03818.pdf), [code](https://github.com/collector-m/RoarNet)  
* 2019 PointRCNN [paper](https://arxiv.org/pdf/1812.04244.pdf), [code](https://github.com/sshaoshuai/PointRCNN)  
* 2019 STD [paper](https://arxiv.org/pdf/1907.10471.pdf). 
* 2019 KpConv [paper](https://arxiv.org/pdf/1904.08889.pdf), [code](https://github.com/HuguesTHOMAS/KPConv)  
* 2019 ShellNet [paper](https://arxiv.org/pdf/1908.06295.pdf), [code](https://github.com/hkust-vgd/shellnet)      
* 2020 SalsaNet [paper](https://arxiv.org/pdf/1909.08291.pdf), [code](https://gitlab.com/aksoyeren/salsanet)  
* 2020 RandLaNet [paper](https://arxiv.org/pdf/1911.11236.pdf), [code](https://github.com/QingyongHu/RandLA-Net)  
* 2020 Part-A2 [paper](https://arxiv.org/pdf/1907.03670.pdf), [code](https://github.com/open-mmlab/OpenPCDet)  
* 2020 3DSSD [paper](https://arxiv.org/pdf/2002.10187.pdf), [code](https://github.com/dvlab-research/3DSSD)  
* 2020 SASSD [paper](https://www4.comp.polyu.edu.hk/~cslzhang/paper/SA-SSD.pdf), [code](https://github.com/skyhehe123/SA-SSD)  
* 2020 CIA-SSD [paper](https://arxiv.org/pdf/2012.03015.pdf), [code](https://github.com/Vegeta2020/CIA-SSD)   
* 2020 Point-GNN [paper](https://arxiv.org/pdf/2003.01251.pdf), [code](https://github.com/WeijingShi/Point-GNN)  
* 2021 S3Net [paper](https://arxiv.org/pdf/2103.08745.pdf).  
* 2021 LSNet [paper](https://www.mdpi.com/2072-4292/14/7/1539).  
* 2021 Pyramid RCNN [paper](https://arxiv.org/pdf/2109.02499.pdf), [code](https://github.com/PointsCoder/Pyramid-RCNN)  
* 2021 ST3D [paper](https://arxiv.org/pdf/2103.05346.pdf), [code](https://github.com/CVMI-Lab/ST3D)  
* 2021 POAT-Net: Parallel Offset-Attention Assisted Transformer for 3D Object Detection for Autonomous Driving. [paper](https://ieeexplore.ieee.org/iel7/6287639/9312710/09611257.pdf). 
* 2022 SASA [paper](https://arxiv.org/pdf/2201.01976.pdf), [code](https://github.com/blakechen97/SASA)  
* 2022 PointDistiller [paper](https://arxiv.org/pdf/2205.11098.pdf), [code](https://github.com/RunpeiDong/PointDistiller)  

**Projection-based**  
* 2015 MVCNN [paper](), [code]()   
* 2017 DepthCN [paper](), [code]()   
* 2017 MV3D [paper](), [code]()   
* 2018 GVCNN [paper](), [code]()  
* 2018 PointSeg [paper](), [code]()  
* 2018 SnapNet [paper](), [code]()  
* 2018 SqueezeSeg [paper](), [code]()  
* 2018 RT3D [paper](), [code]()   
* 2018 BirdNet [paper](), [code]()   
* 2018 PIXOR [paper](), [code]()   
* 2018 Complex-YOLO [paper](), [code]()   
* 2019 SqueezeSeg v2 [paper](), [code]()  
* 2020 BirdNet+ [paper](), [code]()    
* 2020 MVL Net [paper](), [code]()   
* 2020 RAANet [paper](), [code]()   
* 2022 Pseudo-L [paper](), [code]()   
* 2020 SRDL [paper](), [code]()   
* 2023 Ri-fusion [paper](), [code]()  

**Voxel-based**  
* 2015 VoxNet [paper](), [code]()  
* 2017 3DFCN [paper](), [code]()  
* 2017 SegCloud [paper](), [code]()   
* 2017 octNet [paper](), [code]()  
* 2017 Kd-Net [paper](), [code]()  
* 2018 VoxelNet [paper](), [code]()  
* 2018 3DmFV [paper](), [code]()  
* 2018 PointGrid [paper](), [code]()  
* 2018 O-CNN [paper](), [code]()  
* 2019 MVX-Net [paper](), [code]()   
* 2019 F-pointNet [paper](), [code]()  
* 2020 PVRCNN [paper](), [code]()   
* 2020 SECOND [paper](), [code]()  
* 2020 HVNet [paper](), [code]()  
* 2020 Associate-3Ddet [paper](), [code]()   
* 2021 VoxelRCNN [paper](), [code]()   
* 2021 TANet [paper](), [code]()   
* 2021 VIN [paper](), [code]()  
* 2022 SIENet [paper](), [code]()   
* 2022 SMSNet [paper](), [code]()   
* 2022 MA-MFFC [paper](), [code]()   
* 2022 PDV [paper](), [code]()   
* 2023 SAT-GCN [paper](), [code]()    

**Pillar-based**  
* 2019 PointPillars [paper](), [code]()   
* 2022 ASCNet [paper](), [code]()    
* 2022 WCNN3D [paper](), [code]()   

**Hybrid-based**  
* 2020 SegVoxelNet [paper](), [code]()     
* 2020 AMVnet [paper](), [code]()     
* 2020 FusionNet [paper](), [code]()     
* 2021 2-S3Net [paper](), [code]()     
* 2021 JS3C-Net [paper](), [code]()    


