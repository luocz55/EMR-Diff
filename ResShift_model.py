import os
from copy import deepcopy
import numpy as np
import torch
import tqdm
import math
import cv2
import torch.nn.functional as F
from arch.BAFUnet import  BAFUNet
from dataset_loader.dataloader import DataloaderSimpleTest, DataloaderSimpleTrain
from torch.utils.data import DataLoader
from torchmetrics.image import *
import scipy.io as sio
from tqdm import tqdm
from EMRDiff import *
import time
from thop import profile

def sparse_checkerboard(img,size):
    b,c,h,w = img.shape
    img_array = np.array(img.detach().cpu().numpy())
    result_array = np.zeros_like(img_array)
    height, width = h,w
    for i in range(0, height, size):
        for j in range(0, width, size):
                result_array[:, :,i, j] = img_array[:, :,i, j]
    return torch.from_numpy(result_array)
def pdown(img,size):
    b,c,h,w = img.shape
    img_array = np.array(img.detach().cpu().numpy())
    height, width = h,w
    new_height = height // size
    new_width = width // size
    compact_array = np.zeros((b,c,new_height, new_width), dtype=img_array.dtype)
    for i in range(0, height, size):
        for j in range(0, width, size):
            if i // size < new_height and j // size < new_width:
                compact_array[:, :, i // size, j // size] = img_array[:,:,i, j]
    return torch.from_numpy(compact_array)
def save_checkpoint(model, epoch, data):
        model_out_path = "checkpoints/{}_{}/model_epoch_{}.pth.tar".format('EDRDIFF', data, epoch)
        state = {"epoch": epoch, "model": model}

        if not os.path.exists("checkpoints/{}_{}_{}".format('EDRDIFF', data, epoch)):
            os.makedirs("checkpoints/{}_{}_{}".format('EDRDIFF', data,epoch))

        torch.save(state, model_out_path)

        print("Checkpoints saved to {}".format(model_out_path))
class ResShiftTrainer:
    def __init__(self, configs):
        self.configs = configs
        self.device = torch.device("cuda:1")
        self.epochs = self.configs.train['epochs']
        self.num_timesteps = self.configs.diffusion.params.get("steps")
        self.diffusion_sf = self.configs.diffusion.params.get("sf")
        self.diffusion_scale_factor = self.configs.diffusion.params.get("scale_factor")

        self.train_dataloader = self.build_training_dataloader()
        self.val_dataloader = self.build_val_dataloader()
        self.build_model()
        self.build_diffusion_model()
        self.setup_optimization()
        self.psnrall = 0
        self.samall = 0
        self.ssimall = 0
        self.ergasall = 0
    def setup_optimization(self):
        self.optimizer = torch.optim.Adam(self.Net.parameters(), lr=self.configs.train.get('lr'))
    def build_model(self):
        params = self.configs.model.get('params', dict)
        self.Net = BAFUNet(**params)
        self.Net = self.Net.to(self.device)
    def build_diffusion_model(self):
        diffusion_opt = self.configs.get('diffusion', dict)
        self.EMRDIFF = EMRDIFF(diffusion_opt)
    def build_training_dataloader(self):
        opt = {}
        opt['paths'] = self.configs.data.train.params['dir_paths']
        opt['sf'] = self.configs.diffusion.params.get("sf")
        opt['gt_size'] = self.configs.data.train.params.get('gt_size')
        batch_size = self.configs.train.get('batch')[0]
        num_workers = self.configs.train.get('num_workers')
        return DataLoader(DataloaderSimpleTrain(opt), batch_size=batch_size, shuffle=True,num_workers=num_workers)
    def build_val_dataloader(self):
        opt = {}
        opt['paths'] = self.configs.data.val.params['dir_paths']
        opt['sf'] = self.configs.diffusion.params.get("sf")
        opt['gt_size'] = self.configs.data.train.params.get('gt_size')
        batch_size = self.configs.train.get('batch')[1]
        num_workers = self.configs.train.get('num_workers')
        return DataLoader(DataloaderSimpleTest(opt), batch_size=batch_size, shuffle=False,num_workers=num_workers)


    def train(self, epoch, verbose):
        for i in range(epoch):
            i = i
            for step, [gt, lq, rgb] in enumerate(tqdm(self.train_dataloader)):

                lq_up8 = nn.functional.interpolate(lq, scale_factor=8, mode='bicubic', align_corners=False)
                lq_up4 = nn.functional.interpolate(lq, scale_factor=4, mode='bicubic', align_corners=False)
                lq_up2 = nn.functional.interpolate(lq, scale_factor=2, mode='bicubic', align_corners=False)
                lq_up8 = lq_up8.to(self.device).type(torch.float32)
                lq_up4 = lq_up4.to(self.device).type(torch.float32)
                lq_up2 = lq_up2.to(self.device).type(torch.float32)
                lq = lq.to(self.device).type(torch.float32)
                rgb = rgb.to(self.device).type(torch.float32)
                gt = gt.to(self.device).type(torch.float32)
                lqrgb = torch.cat((lq_up8, rgb), dim=1)
                tt = torch.randint(
                    0, self.num_timesteps,
                    size=(lq.shape[0],),
                    device=lq.device,
                )
                noise = torch.randn(
                    size=lqrgb.shape,
                    device=lq.device,
                )
                band = [0, 1, 2]
                self.optimizer.zero_grad()
                p_MSI = gt[:, band, :, :]
                x_start = torch.cat((gt,p_MSI), dim=1)
                x_t = self.EMRDIFF.forward_addnoise(x_start=x_start, y=lqrgb, t=tt, noise=noise,
                                                                    rgb_hr=rgb)
                network_output, up_out = self.Net(x_t, rgb, lq_up8, tt)
                loss_func = nn.L1Loss()
                rgb_down8 = pdown(rgb, 8).to(self.device)
                rgb_down4 = pdown(rgb, 4).to(self.device)
                rgb_down2 = pdown(rgb, 2).to(self.device)
                x_down8 = pdown(x_start, 8).to(self.device)
                x_down4 = pdown(x_start, 4).to(self.device)
                x_down2 = pdown(x_start, 2).to(self.device)
                lqrgb_64 = torch.cat((lq, rgb_down8), dim=1)
                lqrgb_128 = torch.cat((lq_up2, rgb_down4), dim=1)
                lqrgb_256 = torch.cat((lq_up4, rgb_down2), dim=1)
                loss1 = loss_func(network_output + lqrgb, x_start)
                loss2 = loss_func(up_out[2] + lqrgb_64, x_down8)
                loss3 = loss_func(up_out[4] + lqrgb_128, x_down4)
                loss4 = loss_func(up_out[6] + lqrgb_256, x_down2)
                loss = loss1 + loss3 + loss4 + loss2
                loss.backward()
                self.optimizer.step()
                print('epoch: {}'.format(i))
            if (i + 1) % verbose == 0:
                img_index = 0
                for step, [gt, lq, rgb] in enumerate(tqdm(self.val_dataloader)):
                    lq_up8 = nn.functional.interpolate(lq, scale_factor=8, mode='bicubic', align_corners=False)
                    lq_up8 = lq_up8.to(self.device).type(torch.float32)
                    rgb = rgb.to(self.device).type(torch.float32)
                    gt = gt.to(self.device).type(torch.float32)
                    lqrgb = torch.cat((lq_up8, rgb), dim=1)
                    edgeget = Edge()
                    rgb_edge = edgeget(rgb)
                    indices = list(range(self.num_timesteps))[::-1]
                    noise = torch.randn_like(lqrgb)
                    x_t = self.EMRDIFF.prior_sample(lqrgb, noise,edge_map=rgb_edge)
                    psnr = PeakSignalNoiseRatio().to(self.device)
                    sam = SpectralAngleMapper().to(self.device)
                    ssim = MultiScaleStructuralSimilarityIndexMeasure().to(self.device)
                    ergas = ErrorRelativeGlobalDimensionlessSynthesis().to(self.device)
                    results = []
                    for t in indices:
                        tt = torch.tensor([t] * x_t.shape[0], device=x_t.device)
                        with torch.no_grad():
                            lqrgb = torch.cat((lq_up8, rgb), dim=1)
                            x_pred,_ = self.Net(x_t, rgb, lq_up8, tt)
                            x_pred = x_pred + lqrgb
                            noise = torch.randn_like(x_pred)
                            x_t = self.EMRDIFF.inverse_denoise(x_start=x_pred, x_t=x_t, t=tt,noise=noise,edge_map=rgb_edge)#
                            results.append(deepcopy(x_t))

                    for j in range(len(results)):
                        img = results[j]
                        results[j] = img.clamp(min=-1.0, max=1.0)
                    results.append(lq)
                    final_prediction = results[-2]
                    band = self.configs.model.params.get('lq_channels', dict)
                    final_prediction = final_prediction[:,0:band-3,:,:]
                    psnr = psnr(final_prediction, gt)
                    ssim = ssim(final_prediction, gt)
                    sam = sam(final_prediction, gt)
                    ergas = ergas(final_prediction, gt)
                    print('PSNR: {:.4f}, ssim: {:.4f}, sam: {:.4f}, ergas:{:.4f}.'.format(psnr, ssim, sam, ergas))
                    mat_data = {'data': final_prediction.squeeze(0).detach().cpu().numpy()}
                    sio.savemat(f'xiaorong/{img_index}.mat', mat_data)
                    save_checkpoint(self.Net, i,'harvard')
                    img_index += 1
                    
    def test(self):
        img_index = 0
        psnr1 = 0
        ssim1 = 0
        sam1 = 0
        ergas1 = 0
        for step, [gt, lq1, rgb] in enumerate(tqdm(self.val_dataloader)):
            lq = nn.functional.interpolate(lq1, scale_factor=8, mode='bicubic', align_corners=False)
            lq = lq.to(self.device).type(torch.float32)
            rgb = rgb.to(self.device).type(torch.float32)
            edgeget = Edge()
            rgb_edge = edgeget(rgb)
            rgb_edge = rgb_edge.to(self.device).type(torch.float32)
            gt = gt.to(self.device).type(torch.float32)
            band_p = [0, 1, 2]
            p_MSI = gt[:, band_p, :, :]
            lqrgb = torch.cat((lq, rgb), dim=1)
            model = self.Net
            checkpoint = torch.load('/home/yaost/ResDiffusion/checkpoints/EDRDIFF_harvard/model_epoch_49.pth.tar', weights_only=False)
            model.load_state_dict(checkpoint['model'].state_dict())
            t = torch.tensor([1]).to(self.device)
            model.eval()

            model_copy = deepcopy(model)
            model_copy.eval()
            flops, params = profile(model_copy, inputs=(lqrgb,lq,rgb,t), verbose=False)
            print(f'FLOPs: {flops}')
            print(f'Params: {params}')
            del model_copy

            lqrgb = torch.cat((lq, rgb), dim=1)
            band_p = self.configs.model.params.get('lq_channels', dict)
            indices = list(range(self.num_timesteps))[::-1]
            noise = torch.randn_like(lqrgb)
            x_t = self.EMRDIFF.prior_sample(lqrgb, noise, edge_map=rgb_edge)
            psnr = PeakSignalNoiseRatio().to(self.device)
            sam = SpectralAngleMapper().to(self.device)
            ssim = MultiScaleStructuralSimilarityIndexMeasure().to(self.device)
            ergas = ErrorRelativeGlobalDimensionlessSynthesis().to(self.device)
            results = []
            start_time = time.time()
            for t in indices:
                tt = torch.tensor([t] * x_t.shape[0], device=x_t.device)
                with torch.no_grad():
                    lrrgb = torch.cat((lq, rgb), dim=1)
                    x_pred, _ = model(x_t, rgb, lq, tt)
                    x_pred = x_pred + lrrgb
                    noise = torch.randn_like(x_pred)
                    x_t = self.EMRDIFF.inverse_denoise(x_start=x_pred, x_t=x_t, t=tt,
                                                                       noise=noise, edge_map=rgb_edge)
                    results.append(deepcopy(x_t))
            Ours_time = time.time() - start_time
            print(Ours_time)
            for i in range(len(results)):
                img = results[i]
                results[i] = img.clamp(min=-1.0, max=1.0)
            results.append(lq)
            final_prediction = results[-2]  # 因为最后一个是lq2，所以取倒数第二个
            final_prediction = final_prediction[:, 0:band_p - 3, :, :]
            psnr = psnr(final_prediction, gt)
            ssim = ssim(final_prediction, gt)
            sam = sam(final_prediction, gt)
            ergas = ergas(final_prediction, gt)
            print('PSNR: {:.4f}, ssim: {:.4f}, sam: {:.4f}, ergas:{:.4f}.'.format(psnr, ssim, sam, ergas))
            psnr1 += psnr
            ssim1 += ssim
            sam1 += sam
            ergas1 += ergas
            mat_data = {'data': final_prediction.squeeze(0).detach().cpu().numpy()}
            sio.savemat(f'xiaorong/{img_index}.mat', mat_data)
            img_index += 1
        print('PSNR: {:.4f}, ssim: {:.4f}, sam: {:.4f}, ergas:{:.4f}.'.format(psnr1/10, ssim1/10, sam1/10, ergas1/10))

