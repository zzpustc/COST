import copy
import random
from abc import abstractmethod
from typing import Dict, List, Tuple, Union

import cvxpy as cp
import numpy as np
import torch
import torch.nn.functional as F
from scipy.optimize import minimize, least_squares

import peft
from copy import deepcopy

from experiments.nyuv2.utils import AverageMeter
import pdb

# calculate single batch loss.
def cal_dataset_loss(opt, batch, lora_copy):
    if opt.dataset == 'city':
        from experiments.cityscapes.trainer import calc_loss
        lora_data, lora_label, lora_depth = batch
        lora_data, lora_label = lora_data.cuda(), lora_label.long().cuda()
        lora_depth = lora_depth.cuda()

        lora_pred, _ = lora_copy(lora_data, return_representation=True)
        losses = torch.stack(
            (
                calc_loss(lora_pred[0], lora_label, "semantic"),
                calc_loss(lora_pred[1], lora_depth, "depth"),
            )
        )

    elif opt.dataset == 'nyuv2':
        from experiments.nyuv2.trainer import calc_loss

        lora_data, lora_label, lora_depth, lora_normal = batch
        lora_data, lora_label = lora_data.cuda(), lora_label.long().cuda()
        lora_depth, lora_normal = lora_depth.cuda(), lora_normal.cuda()

        lora_pred, _ = lora_copy(lora_data, return_representation=True)
        losses = torch.stack(
            (
                calc_loss(lora_pred[0], lora_label, "semantic"),
                calc_loss(lora_pred[1], lora_depth, "depth"),
                calc_loss(lora_pred[2], lora_normal, "normal"),
            )
        )

    elif opt.dataset == 'celeba':
        loss_fn   = torch.nn.BCELoss()
        x, y = batch
        x = x.cuda()
        y = [y_.cuda() for y_ in y]
        y_ = lora_copy(x)
        losses = torch.stack([loss_fn(y_task_pred, y_task) for (y_task_pred, y_task) in zip(y_, y)])

    elif opt.dataset == 'qm9':
        data = batch.cuda()
        out, _ = lora_copy(data, return_representation=True)
        losses = F.mse_loss(out, data.y, reduction="none").mean(0)

    return losses

# calculate avg loss of train loader before tele.
def obtain_loss_contour(opt, multi_task_model, lora_loader):

    multi_task_model.eval()
    if opt.dataset == 'city':
        loss_backup = [0] * 2
    elif opt.dataset == 'nyuv2':
        loss_backup = [0] * 3
    elif opt.dataset == 'celeba':
        loss_backup = [0] * 40
    elif opt.dataset == 'qm9':
        loss_backup = [0] * 11

    for j, data in enumerate(lora_loader):  
        losses = cal_dataset_loss(opt, data, multi_task_model)
        for jj in range(len(losses)):
            loss_backup[jj] += losses[jj].item()    
    for jj in range(len(losses)):
        loss_backup[jj] /= (j+1)

    return loss_backup

# obtain the tensor of model difference between bef. and aft. tele.
def model_diff(m_pre, m_pro):
    grads = torch.cat([torch.flatten(qq - pp) for pp, qq in zip(m_pre.shared_parameters(), m_pro.shared_parameters())]) 
    
    return grads

# sharpness estimation func
def sharpness_estimation(opt, model, cur_data, L0, lora_sign, ratio_list, noise_norm=1e-4):
    sharp_loss = []
    L0 = cal_dataset_loss(opt, cur_data, model)
    for _ in range(opt.grad_sampling_times):
        lora = deepcopy(model)
        if lora_sign:
            lora = lora.merge_and_unload()

        # lora.eval() 
        for param in lora.parameters():
            # Generate random noise
            noise = torch.randn_like(param)
            # Scale noise to have the desired norm
            noise = noise * (noise_norm / torch.norm(noise))
            # Add noise to the parameter
            param.data = param.data + noise

        lora_loss = cal_dataset_loss(opt, cur_data, lora)
        n_tasks = len(lora_loss)
        sharp_loss.append(torch.stack([ratio_list[i] * torch.abs(lora_loss[i] - L0[i]) for i in range(n_tasks)]))

        del lora

    return torch.max(torch.stack(sharp_loss), dim=0)[0].mean() / noise_norm


# sharpness estimation func
def gradient_estimation(opt, model, cur_data, L0, lora_sign, noise_norm=1e-4):
    grad_est = []
    #L0 = cal_dataset_loss(opt, cur_data, model)
    for _ in range(opt.grad_sampling_times):
        lora = deepcopy(model)
        if lora_sign:
            lora = lora.merge_and_unload()

        # lora.eval() 
        for param in lora.shared_parameters():
            # Generate random noise
            noise = torch.randn_like(param)
            # Scale noise to have the desired norm
            noise = noise * (noise_norm / torch.norm(noise))
            # Add noise to the parameter
            param.data = param.data + noise

        lora_loss = cal_dataset_loss(opt, cur_data, lora)
        grad_norm_tmp = lora_loss.mean() #/ noise_norm

        grad_est.append(grad_norm_tmp)

        del lora

    return torch.stack(grad_est).mean()


# main func of teleportation
def teleporation(opt, multi_task_model, lora_loader, lora_config, cur_data, loss_backup, ratio_list):

    multi_task_model.eval()
    print("---Start teleporation with LoRA---")
    print("Obtain current loss contour...")
    loss_backup = obtain_loss_contour(opt, multi_task_model, lora_loader)
    print("Loss contour obtained...")
    # peft test
    lora_copy = peft.get_peft_model(multi_task_model, lora_config)
    lora_optim = torch.optim.Adam(lora_copy.parameters(), lr= opt.lr)
                    
    # start lora training
    for nn in range(opt.tele_round): # training rounds
        loss_record = AverageMeter('Loss', ':.4e')
        task_loss_record = AverageMeter('Task_Loss', ':.4e')
        sharp_loss_record = AverageMeter('Sharp_Loss', ':.4e')
        for j, batch in enumerate(lora_loader):  
        
            losses = cal_dataset_loss(opt, batch, lora_copy)

            n_tasks = len(losses)
            task_obj = torch.mean(torch.stack([torch.abs(losses[ii] - loss_backup[ii]) for ii in range(n_tasks)]))   
                            
            # estimate the gradient via sharpness estimation
            sharp_obj = sharpness_estimation(opt, lora_copy, cur_data, losses, True, ratio_list)
            lora_obj = 10 * task_obj - 0.5 * sharp_obj

            loss_record.update(lora_obj.item(), opt.batch_size)
            task_loss_record.update(task_obj.item(), opt.batch_size)
            sharp_loss_record.update(sharp_obj.item(), opt.batch_size)
            
            if j % 100 == 0:
                print('Inner Epoch: {:04d} | Total Loss: {:.2f} (Ave.: {:.2f}) | Task Loss: {:.2f} (Ave.: {:.2f}) | Sharp Loss: {:.2f} (Ave.: {:.2f})'.format(nn, lora_obj.item(), loss_record.avg, task_obj.item(), task_loss_record.avg, sharp_obj.item(), sharp_loss_record.avg))

            lora_optim.zero_grad()
            lora_obj.backward()
            lora_optim.step()


    multi_task_model = lora_copy.merge_and_unload()
    multi_task_model.enable_grad()
    multi_task_model.train()

    del lora_copy, lora_optim
    print("---End of teleporation---")

    return multi_task_model