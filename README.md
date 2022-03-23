
# Skoltech-ML-2022-Drone-Collision-Avoidance-In-Indoor-Environment

The main goal of this study is to demonstrate the approach of achieving drone dynamic and static collision avoidance in an indoor environment using deep neural networks. The system will consist of a swarm of drones, a camera mounted above them, and static obstacles. Images from the camera are segmented into three groups: drones, obstacles, and floor. For the task of semantic
segmentation, manual annotation is needed for our small custom dataset, which we use for training several neural networks. Computer vision algorithms process the segmented image and return the coordinates of the obstacle relative to the drone. Then we train the RL NN on the coordinates of the drones and static collisions and get the possible safe actions of the drones in real-time. We evaluate deep learning models trained on both synthetic and real data and present a new dataset that comprises both.

<img src="https://github.com/makmary/Skoltech-ML-2022-Drone-Collision-Avoidance-In-Indoor-Environment/blob/main/ezgif.com-gif-maker.gif">

The code was written by:

- Mariia Makarova - U-net model implementation
- Ayush Gupta - Semantic Image Segmentation
- Ahmed Baza - Drone Simulation
- Ekaterina Dorzhieva - RL based Path Planning

**Drone collision avoidance in indoor environment: [Project](https://github.com/makmary/Skoltech-ML-2022-Drone-Collision-Avoidance-In-Indoor-Environment) | [Report]() | [Presentation]() | [Video](https://drive.google.com/drive/folders/1iRLgcNHrFjxwGGnlAggEWa8eK2ZidKMn?usp=sharing)**


## Notebooks
- Semantic Image Segmentation: [notebook](https://github.com/makmary/Skoltech-ML-2022-Drone-Collision-Avoidance-In-Indoor-Environment/blob/main/semantic-segmentation-with-unet/FinalProject.ipynb) 
- RL Based Collision Avoidance: [notebook](https://github.com/makmary/Skoltech-ML-2022-Drone-Collision-Avoidance-In-Indoor-Environment/blob/main/foraging-v0-master/Training-for-RL.ipynb) | [README](https://github.com/makmary/Skoltech-ML-2022-Drone-Collision-Avoidance-In-Indoor-Environment/blob/main/foraging-v0-master/README_RL.md)
- Environment Visualization (testing in simulated environment): [notebook](https://github.com/makmary/Skoltech-ML-2022-Drone-Collision-Avoidance-In-Indoor-Environment/blob/main/foraging-v0-master/Environment_visualisation.ipynb) | [README](https://github.com/makmary/Skoltech-ML-2022-Drone-Collision-Avoidance-In-Indoor-Environment/blob/main/foraging-v0-master/README_vis.md)
## Prerequisites
- Python 3
- Segmentation model
- Tensorflow 2.8.0, Keras 2.8.0

## Datasets info
- Our custom dataset:  [dataset for drones](https://github.com/makmary/Skoltech-ML-2022-Drone-Collision-Avoidance-In-Indoor-Environment/tree/main/semantic-segmentation-with-unet/data)

## How to launch the code?
To help users better understand and use our code, for each model we created instructions for running the code and reproducing the results:

-  instruction for running the code and reproducing the results of Image Segmentation: [Image Segmentation Instruction](https://github.com/makmary/Skoltech-ML-2022-Drone-Collision-Avoidance-In-Indoor-Environment/blob/main/semantic-segmentation-with-unet/README.md)

-  instruction for running the code and reproducing the results of RL Based PP: [RL based Path Planning Instruction 1](https://github.com/makmary/Skoltech-ML-2022-Drone-Collision-Avoidance-In-Indoor-Environment/blob/main/foraging-v0-master/README_RL.md) | [RL based Path Planning Instruction 2](https://github.com/makmary/Skoltech-ML-2022-Drone-Collision-Avoidance-In-Indoor-Environment/blob/main/foraging-v0-master/README_vis.md)


