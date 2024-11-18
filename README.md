# Robust Brain Tumor Segmentation with Incomplete MRI Modalities Using Hölder Divergence and Mutual Information-Enhanced Knowledge Transfer
Official implementatation for paper: Robust Brain Tumor Segmentation with Incomplete MRI Modalities Using Hölder Divergence and Mutual Information-Enhanced Knowledge Transfer

## Environment
The required libraries are listed in `environment.yml`
```
cond create -n your_name -f environment.yml
```
## Data preparation
download [BraTS18](https://www.med.upenn.edu/sbia/brats2018/registration.html) and modify paths in `mypath.py`

## training & eval
run `sh cli/train.sh`
