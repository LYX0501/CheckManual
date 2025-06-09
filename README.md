# CheckManual: A New Challenge and Benchmark for Manual-based Appliance Manipulation

we propose the first manual-based appliance manipulation benchmark **CheckManual**. Specifically, we design a large model-assisted human-revised data generation pipeline to create manuals based on CAD appliance models. With these manuals, we establish novel manual-based manipulation challenges, metrics, and simulator environments for model performance evaluation. Furthermore, we propose the first manual-based manipulation planning model **ManualPlan** to set up a group of baselines for the  **CheckManual** benchmark.

<p align="center">
  <img src="images/Teasor.jpg" style="width:80%;">
</p>



## 🔥 News
- 2025.06.09: We have released the ManualPlan framework and evaluation script for Track 1 challenge.
- 2025.06.05: [**CheckManual**](https://drive.google.com/file/d/1YasM5Se7h4H8wCqZFN3mK8sCu1cEZBo7/view?usp=drive_link) dataset has been released.
- 2025.04.04: Our paper is announced as CVPR 2025 **Highlight**.
- 2025.02.26: Our paper [**CheckManual: A New Challenge and Benchmark for Manual-based Appliance Manipulation**](https://openaccess.thecvf.com/content/CVPR2025/papers/Long_CheckManual_A_New_Challenge_and_Benchmark_for_Manual-based_Appliance_Manipulation_CVPR_2025_paper.pdf) is accepted by CVPR 2025. 

## 🌏 Environment

### Data Preparation
Please download the [**PartNet-Mobility**](https://sapien.ucsd.edu/downloads) dataset and the [**CheckManual**](https://drive.google.com/file/d/1YasM5Se7h4H8wCqZFN3mK8sCu1cEZBo7/view?usp=drive_link) dataset.

Then, you should rearrange them in the **data** file as the following format.

```
|data
| -- sapien_dataset
|    | -- 148
|    | -- 149
|    | -- 152
|    `-- ...
| -- checkmanual_dataset
|    | -- manual_1
|    | -- manual_2
|    | -- manual_3
|    `-- ...
```

### Installation
We have tested the following installation steps on the [**AutoDL**](https://www.autodl.com/market/list) RTX 3090 workstation with Ubuntu 20.04 and CUDA 11.3.

First, create Conda environment
```
conda create -n checkmanual python=3.7
conda activate checkmanual
pip install torch==1.13.1+cu116 torchvision==0.14.1+cu116 torchaudio==0.13.1 --extra-index-url
sudo apt update
sudo apt install xvfb poppler-utils
```
Git clone CheckManual repository
```
git clone https://github.com/LYX0501/CheckManual.git
cd CheckManual
```
Then, install SAPIEN (Python 3.7) following
```
pip install http://download.cs.stanford.edu/orion/where2act/where2act_sapien_wheels/sapien-0.8.0.dev0-cp37-cp37m-manylinux2014_x86_64.whl
```
For other Python versions, you can use one of the following
```
pip install http://download.cs.stanford.edu/orion/where2act/where2act_sapien_wheels/sapien-0.8.0.dev0-cp35-cp35m-manylinux2014_x86_64.whl
pip install http://download.cs.stanford.edu/orion/where2act/where2act_sapien_wheels/sapien-0.8.0.dev0-cp36-cp36m-manylinux2014_x86_64.whl
pip install http://download.cs.stanford.edu/orion/where2act/where2act_sapien_wheels/sapien-0.8.0.dev0-cp38-cp38-manylinux2014_x86_64.whl
```
Please do not use the default `pip install sapien` as SAPIEN is being actively updated.

You also needs to install other packages by executing
```
pip install -r requirements.txt
```

### Configure GPT and OCR API
Before calling GPT and OCR, you need to configure their keys in `api_utils/api_key_config.json` file.

In our work, we use GPT API provided by [**ChatAnyWhere**](https://api.chatanywhere.org/#/) and OCR API provided by [**Baidu**](https://ai.baidu.com/tech/ocr).

### Track 1: Run Evaluation about ManualPlan
You run the evaluation about ManualPlan on Track 1 challenge by:
```
xvfb-run -a python track1_ManualPlan.py
```
This python script will create `track1_result.json` file to record the evaluation results.

## ✒ Citation
Please cite our paper if you find it helpful :)
```
@article{checkmanual,
    author    = {Long, Yuxing and Zhang, Jiyao and Pan, Mingjie and Wu, Tianshu and Kim, Taewhan and Dong, Hao},
    title     = {CheckManual: A New Challenge and Benchmark for Manual-based Appliance Manipulation},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2025},
}
```
