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
| KITTI    |   22  |    8    | 15k | 200k |   CL   | Germany | ------|
|nuScene   |1k     | 23      | 40k | 1.4M | CL    |Boston & Singapore| --- |
|Waymo Open| 1k    | 4       | 200k | 12M | CL | Different cities| ----|
| Apollo scape| - | 35 | 130k| 70k | CL | 10 cities in China | -|
| City Scape| - | 30| 30k| CL| 50 cities| - |
|Oxford Robot Car| -| - | - | CLR| Oxford UK | - |
|H3D|160|8|27K|1.07M|L|San Francisco Bay| -|
|Lyft level 5| -|-|-|-|CL| Palo Alto, California|-| 
|Panda sets| 8|28|80k|-|CL|-|-|
|ONCE|1M|5|15k|-|CL|china||-|
|AIODrive|-|-|-|10M|CLR|-|-|
|Ford Multi-AV| -|-|-|-|CL|Michigan USA|
|DDAD|-|-|-|-|CL|USA & Japan|-|
|CADC|75|10|7k|-|CL|Waterloo Canada| -|
|A2D2|-|-|12k|-|CL|Germany|-|
|Ax3D|-|7|230k|-|CL|Singapore|-|
|Argoverse|1K|-|-|-|CL|Pittdburg & Miami USA|-|
|BLVD|654|3|249k|-|CL|Changshu China|-|
|Itacha 365|40|6|-|-|CL|Itacha, New York|-|
|Rope3D|26|12|-|1.5M|CL|-|-|

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



