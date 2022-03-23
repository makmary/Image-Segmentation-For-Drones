
# Semantic Segmentation For Drones

The main goal of this study is to demonstrate the approach of achieving drone dynamic and static collision avoidance in an indoor environment using deep neural networks. The system will consist of a swarm of drones, a camera mounted above them, and static obstacles. Images from the camera are segmented into three groups: drones, obstacles, and floor. For the task of semantic
segmentation, manual annotation is needed for our small custom dataset, which we use for training several neural networks. Computer vision algorithms process the segmented image and return the coordinates of the obstacle relative to the drone. Then we train the RL NN on the coordinates of the drones and static collisions and get the possible safe actions of the drones in real-time. We evaluate deep learning models trained on both synthetic and real data and present a new dataset that comprises both.

## Notebooks


## Prerequisites
- Python 3, Pandas, Numpy, Random
- Patchify, Imagecodecs, Matplotlib, Tifffile, PIL
- Segmentation models, OpenCV, 
- Tensorflow 2.8.0, Keras 2.8.0, Scipy

## Datasets info
- Our custom dataset:  [dataset for drones](https://github.com/makmary/Skoltech-ML-2022-Drone-Collision-Avoidance-In-Indoor-Environment/tree/main/semantic-segmentation-with-unet/data)


## How to launch the code?
