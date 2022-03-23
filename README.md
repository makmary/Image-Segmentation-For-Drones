
# Skoltech-ML-2022-Drone-Collision-Avoidance-In-Indoor-Environment

The main goal of this study is to demonstrate the approach of achieving drone dynamic and static collision avoidance in an indoor environment using deep neural networks. The system will consist of a swarm of drones, a camera mounted above them, and static obstacles. Images from the camera are segmented into three groups: drones, obstacles, and floor. For the task of semantic
segmentation, manual annotation is needed for our small custom dataset, which we use for training several neural networks. Computer vision algorithms process the segmented image and return the coordinates of the obstacle relative to the drone. Then we train the RL NN on the coordinates of the drones and static collisions and get the possible safe actions of the drones in real-time. We evaluate deep learning models trained on both synthetic and real data and present a new dataset that comprises both.

![project](https://github.com/makmary/Skoltech-ML-2022-Drone-Collision-Avoidance-In-Indoor-Environment)

The code was written by:

- Mariia Makarova - U-net model implementation
- Ayush Gupta - Semantic Image Segmentation
- Ahmed Baza - Drone Simulation
- Ekaterina Dorzhieva - RL based Path Planning

**Drone collision avoidance in indoor environment: [Project](https://github.com/makmary/Skoltech-ML-2022-Drone-Collision-Avoidance-In-Indoor-Environment) | [Report]() | [Presentation]() | [Video]()**


## Notebooks


## Prerequisites
- Python 3
- Segmentation model
- Tensorflow 2.8.0, Keras 2.8.0

## Datasets info
- Our custom dataset:  [dataset for drones](https://github.com/makmary/Skoltech-ML-2022-Drone-Collision-Avoidance-In-Indoor-Environment/tree/main/semantic-segmentation-with-unet/data)


## How to launch the code?
To help users better understand and use our code, for each model we created instructions for running the code and reproducing the results:

-  instruction for running the code and reproducing the results: [Instruction1]()

-  instruction for running the code and reproducing the results: [Instruction2]()


