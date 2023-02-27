# habitat-tools
This repository provides a minimal set of tools for working with the Habitat environment [[1]](#references) in Python. I built these tools when working on research with the Habitat environment. Hopefully they are helpful to other people.

<img src='Figs/title.png'/>

## Dependencies
We use `python==3.7.4`.  
We recommend using a conda environment.  
```
conda create --name habitat_py37 python=3.7.4
source activate habitat_py37
```
You can install Habitat-Lab and Habitat-Sim following instructions from [here](https://github.com/facebookresearch/habitat-lab "here").  
We recommend to install Habitat-Lab and Habitat-Sim from the source code.  
We use `habitat==0.2.2` and `habitat_sim==0.2.2`.  
Use the following commands to set it up:  
```
# install habitat-lab
git clone --branch stable https://github.com/facebookresearch/habitat-lab.git
cd habitat-lab
git checkout tags/v0.2.2
pip install -e .

# install habitat-sim
git clone --recurse --branch stable https://github.com/facebookresearch/habitat-sim.git
cd habitat-sim
pip install -r requirements.txt
sudo apt-get update || true
# These are fairly ubiquitous packages and your system likely has them already,
# but if not, let's get the essentials for EGL support:
sudo apt-get install -y --no-install-recommends \
     libjpeg-dev libglm-dev libgl1-mesa-glx libegl1-mesa-dev mesa-utils xorg-dev freeglut3-dev
git checkout tags/v0.2.2
python setup.py install --with-cuda
```


## Demo 1: build a top-down-view semantic map
```
python demo_build_semantic_BEV_map.py
```
This demo densely sample rendered observations at different locations of the current environment.



## References
[1] Savva, M., Kadian, A., Maksymets, O., Zhao, Y., Wijmans, E., Jain, B., ... & Batra, D. (2019). Habitat: A platform for embodied ai research. In Proceedings of the IEEE/CVF international conference on computer vision (pp. 9339-9347). [https://github.com/facebookresearch/habitat-lab](https://github.com/facebookresearch/habitat-lab)