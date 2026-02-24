from omegaconf import OmegaConf
from model.ResShift_model import ResShiftTrainer
import torch

if __name__ == "__main__":
    print(torch.cuda.is_available())
    path = r"config/realsr_swinunet_realesrgan256_journal_simple.yaml"
    configs = OmegaConf.load(path)      
    Trainer = ResShiftTrainer(configs=configs)
    # Trainer.evaluate() ##测试模型3推理
    Trainer.train(4000,40)  ##重新训练模型

