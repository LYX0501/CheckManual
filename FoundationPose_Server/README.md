## FoundationPose Env Setup
```
git clone https://github.com/NVlabs/FoundationPose.git
cd FouondationPose

# Setup conda environment
conda create -n foundationpose python=3.9
conda activate foundationpose

# Install Eigen3 3.4.0 under conda environment
cd /tmp
wget https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.tar.gz
tar -xvzf eigen-3.4.0.tar.gz
cd eigen-3.4.0
mkdir build && cd build
cmake ..
sudo make install
sudo apt install libboost-all-dev
cd /path/to/FoundationPose

# Install dependencies
python -m pip install -r requirements.txt

# Install NVDiffRast
python -m pip install --quiet --no-cache-dir git+https://github.com/NVlabs/nvdiffrast.git

# Install PyTorch3D
python -m pip install --quiet --no-index --no-cache-dir pytorch3d -f https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py39_cu118_pyt200/download.html

# Build extensions
CMAKE_PREFIX_PATH=$CONDA_PREFIX/lib/python3.9/site-packages/pybind11/share/cmake/pybind11 bash build_all_conda.sh

# Install pyrender(For visualizing predicted pose)
sudo apt install -y libegl1 libglu1-mesa libgl1-mesa-glx

# Install Flask
pip install flask
```

## Download FoundationPose weights
Download all network weights from [here](https://drive.google.com/drive/folders/1DFezOAD0oD1BblsXVxqDsl8fj0qzB82i?usp=sharing) and put them under the folder `weights/`. For the refiner, you will need `2023-10-28-18-33-37`. For scorer, you will need `2024-01-11-20-02-45`.

## Run the FoundationPose Flask Server
```
python foundationpose_flask.py
```
