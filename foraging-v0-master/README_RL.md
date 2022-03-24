# RL based Planning: 
## Dependencies: 
- [numpy](https://pypi.org/project/numpy/)
- [gym](https://pypi.org/project/gym/)
- [collection](https://pypi.org/project/collection/)
- [matplotlib](https://pypi.org/project/matplotlib/)
- [pytorch](https://pytorch.org/get-started/locally/) follow the provided link for installation based on your system. 
- [stl](https://pypi.org/project/stl/) 
- [sklearn](https://scikit-learn.org/stable/install.html)
- [copy](https://pypi.org/project/pycopy-copy/)
### Instal dependencies

``` sh 
pip install numpy
pip install gym
pip install collection 
pip install matplotlib 
pip instal stl 
pip install -U scikit-learn
pip install pycopy-copy
```
To install the package, cd to the root directory of the package (Skoltech-ML-2022-Drone-Collision-Avoidance-In-Indoor-Environment/foraging-v0-master) and run:
``` sh 
pip install -e .
```

## Running instructions: 
- Run the notebook (ML_final_act_crt.ipynb) for training 
- You can load the models whights as they are saved

``` sh 
torch.load(actor.pth)
```
The loading line is commented in the notebook too 
