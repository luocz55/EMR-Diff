# EMR-Diff
CVPR 2026: Edge-aware Multimodal Residual Diffusion Model for Hyperspectral Image Super-resolution

![Static Badge](https://img.shields.io/badge/Paper-CVPR%202026-brightgreen?style=flat)← click here to read the paper~

# Framework
![Paper](EMR-Diff.jpg)

# Installation
python==3.11
omegaconf==2.3.0
tqdm==4.65.2
thop==0.1.1
scipy==1.16.0
torchmetrics==1.7.2
numpy==1.26.4

# Parameter settings
You can adjust the model parameters at `config/5_step_EMRDiff.yaml`
```
data:
  train:
    params:
      dir_paths: ['hardvard']  # Put the training dataset path here
      gt_size: 512
  val:
    params:
      dir_paths: ['hardvardtest'] # Put the testing dataset path here
```

# Train
```python
python Train.py
```
# Test
```python
python Test.py
```

