# EMR-Diff
[![Static Badge](https://img.shields.io/badge/Paper-CVPR%202026-brightgreen?style=flat)](https://openaccess.thecvf.com/content/CVPR2026/papers/Zhang_EMR-Diff_Edge-aware_Multimodal_Residual_Diffusion_Model_for_Hyperspectral_Image_Super-resolution_CVPR_2026_paper.pdf)
← click here to read the paper~
# Installation
```
python==3.11
omegaconf==2.3.0
tqdm==4.65.2
thop==0.1.1
scipy==1.16.0
torchmetrics==1.7.2
numpy==1.26.4
```

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
train:
  lr:  1e-4          # learning rate
  batch: [1,1]       #[train batchsize,test batchsize]
  num_workers: 0
  microbatch: 1
  epochs: 3000       # train epochs
  test_frequency: 100  # Testing frequency during training
```

You can adjust the save location at `model/ResShift_model.py`
```
    sio.savemat(f'xiaorong/{img_index}.mat', mat_data) # Output image saved during testing
    ......
    save_checkpoint(self.Net, i + 1,'harvard') # Save the model of the current epoch
```
Data preprocessing at dataset_loader/dataloader.py
```
Chikusei dataset's SRF can be obtained through SRF/SRF_made.py interpolation
```
# Train
```python
python Train.py
```
# Test
```python
python Test.py
```

