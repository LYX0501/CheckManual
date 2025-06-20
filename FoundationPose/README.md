## Foundationpose Env Setup

```bash
git clone https://github.com/NVlabs/FoundationPose.git
cd FouondationPose

# create conda environment
conda create -n foundationpose python=3.9
# activate conda environment
conda activate foundationpose

# # Install Eigen3 3.4.0 under conda environment
# 按照foundationpose的readme安装，发生了错误
# conda install conda-forge::eigen=3.4.0
# export CMAKE_PREFIX_PATH="$CMAKE_PREFIX_PATH:/eigen/path/under/conda"
cd /tmp
wget https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.tar.gz
tar -xvzf eigen-3.4.0.tar.gz
cd eigen-3.4.0
mkdir build && cd build
cmake ..
sudo make install
sudo apt install libboost-all-dev
cd /path/to/foundationpose

# install dependencies
python -m pip install -r requirements.txt

# Install NVDiffRast
python -m pip install --quiet --no-cache-dir git+https://github.com/NVlabs/nvdiffrast.git

# PyTorch3D
python -m pip install --quiet --no-index --no-cache-dir pytorch3d -f https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py39_cu118_pyt200/download.html

# Build extensions
CMAKE_PREFIX_PATH=$CONDA_PREFIX/lib/python3.9/site-packages/pybind11/share/cmake/pybind11 bash build_all_conda.sh

# pyrender(可视化pose预测结果才需要,不安装的话需要把可视化部分代码删掉)
sudo apt install -y libegl1 libglu1-mesa libgl1-mesa-glx

```

## Foundationpose weights

Download all network weights from here and put them under the folder weights/. For the refiner, you will need 2023-10-28-18-33-37. For scorer, you will need 2024-01-11-20-02-45