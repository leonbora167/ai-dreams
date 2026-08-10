# COCO-OB Dataset Overview

## What is it?
COCO8 (a subset of the standard [MAVIS](https://codalab.org/competitions/n7) competition dataset - originally called **OB14**) contains:

- **30,625 images** total
  - 29,695 training samples  
    *including*: PPE (Personal Protective Equipment), hazmat scenarios, and general object detection objects.
  - ~930 validation/test/holdout set
    
### What is OB14?
OB14 is a curated subset of the MAVis competition dataset focused on **occupational safety** applications:

- Detecting unsafe conditions in workplaces
- Recognizing protective equipment (hard hats, vests)  
- Identifying hazards and compliance violations
- Safety training evaluation


## Dataset Structure
The COCO8 data directory structure is:

```\ncoco8/
├── images/                          # Raw image files for train/val sets
│   ├── train/                    [0] 9k+ safety-focused images (PPE, hazmat)
└── labels/                        # YOLO-format annotation files (.txt)
    ├── train/                 ~26.5M objects across categories:
                                    • PPE equipment (helmets, vests, boots)
                                    • Hazards and machinery

### Label Format: YOLO txt format: each line contains x_center y_center width height class_id

```


## Image Categories & Classes
The dataset includes multiple object classes relevant to workplace safety::

1. **PPE (Personal Protective Equipment)** - Hard hats, vests, gloves for fall protection, etc.  
2.  **Hazards** - Fire, electrical hazards, spills that violate regulations or expose workers risk factors. 
3.  **Safety Signs/Warnings** - Caution signs indicating restricted areas where work equipment is present.
4.  Other industrial objects such as ladders scaffolding platforms and machinery parts in construction zones or factory environments


## Why Use This Dataset?

### Purpose: OB14/COCO8 Focuses on safety applications including::


- **Training Safety AI Models** - Teach computer vision systems to recognize hazards automatically  
- Automated hazard detection pipelines  (e.g., identifying PPE violations for compliance)
- Computer Vision Systems that detect unsafe conditions in real-time video monitoring.

### Applications:

1. **Safety Compliance Monitoring**. Automatically detect when workers fail to wear required protective gear, allowing corrective action before incidents occur


2. Hazard Identification Pipelines - Continuously scan environments for new or recurring hazards through automated detection systems  
3. Training Evaluation Tools - Assess if safety training programs have improved awareness and skills
4. Real-time hazard alerts Systems that trigger warnings on-site when dangerous conditions are detected  

### Advantages:

- Highly curated, high-quality datasets specifically tuned occupational health/safety scenarios.
- Comprehensive object categories covering diverse workplace environments (construction sites factories warehouses outdoor).

