# CheckManual

we propose the first manual-based appliance manipulation benchmark **CheckManual**. Specifically, we design a large model-assisted human-revised data generation pipeline to create manuals based on CAD appliance models. With these manuals, we establish novel manual-based manipulation challenges, metrics, and simulator environments for model performance evaluation. Furthermore, we propose the first manual-based manipulation planning model **ManualPlan** to set up a group of baselines for the  **CheckManual** benchmark.

## 🔥 News
- 2025.4.4: Our paper is announced as CVPR 2025 Highlight.
- 2025.3.28: Our paper has been pended by Arxiv due to unknown problems. We will release codes and data immediately after Arxiv acceptance.
- 2025.2.26: Our paper is accepted by CVPR 2025. 

## Environment

### Download PartNet-Mobility Assets and Our CheckManual Dataset
Please download the PartNet-Mobility dataset from https://sapien.ucsd.edu/downloads and the CheckManual dataset from ....

|data
|----sapien_dataset
|--------148
|--------149
|--------152
|--------......
|....checkmanual_dataset
|--------manual_1
|--------manual_2
|--------manual_3
|--------......


### Install SAPIEN Simulator
First, install SAPIEN following

    pip install http://download.cs.stanford.edu/orion/where2act/where2act_sapien_wheels/sapien-0.8.0.dev0-cp36-cp36m-manylinux2014_x86_64.whl

For other Python versions, you can use one of the following

    pip install http://download.cs.stanford.edu/orion/where2act/where2act_sapien_wheels/sapien-0.8.0.dev0-cp35-cp35m-manylinux2014_x86_64.whl
    pip install http://download.cs.stanford.edu/orion/where2act/where2act_sapien_wheels/sapien-0.8.0.dev0-cp37-cp37m-manylinux2014_x86_64.whl
    pip install http://download.cs.stanford.edu/orion/where2act/where2act_sapien_wheels/sapien-0.8.0.dev0-cp38-cp38-manylinux2014_x86_64.whl

Please do not use the default `pip install sapien` as SAPIEN is being actively updated.


