from omegaconf import OmegaConf
from model.ResShift_model import ResShiftTrainer
import torch

if __name__ == "__main__":
    print(torch.cuda.is_available())
    path = r"config/5_step_EMRDiff.yaml"
    configs = OmegaConf.load(path)      
    Trainer = ResShiftTrainer(configs=configs)
    Trainer.train(2000,200)  #verboser : test frequency

