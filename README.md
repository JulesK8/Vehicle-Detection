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

**- Appearance-based methods**  
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

**e. Underneath shadow-based methods**
* [Vehicle detection based on underneath vehicle shadow using edge features](https://ieeexplore.ieee.org/abstract/document/7893608/)  
* [Shadow-based vehicle detection in urban traffic](https://www.mdpi.com/193936)  
* [Shadow Based On-Road Vehicle Detection and Verification Using HAAR Wavelet Packet Transform](https://ieeexplore.ieee.org/abstract/document/1598621/)  
* [Shadow detection in camera-based vehicle detection: survey and analysis](https://www.spiedigitallibrary.org/journals/Journal-of-Electronic-Imaging/volume-25/issue-5/051205/Shadow-detection-in-camera-based-vehicle-detection--survey-and/10.1117/1.JEI.25.5.051205.short?SSO=1)  

**f. Texture-based methods**
* [Robust vehicle detection in vision systems based on fast wavelet transform and texture analysis](https://ieeexplore.ieee.org/abstract/document/4339088/)  
* [Real-time small obstacle detection on highways using compressive RBM road reconstruction](https://ieeexplore.ieee.org/abstract/document/7225680/) 

**- Stereo-vision-based methods**  
* [GOLD: A parallel real-time stereo vision system for generic obstacle and lane detection](https://ieeexplore.ieee.org/abstract/document/650851/)  
* [Vehicle detection based on multifeature extraction and recognition adopting RBF neural network on ADAS system](https://www.hindawi.com/journals/complexity/2020/8842297/)  
* 



