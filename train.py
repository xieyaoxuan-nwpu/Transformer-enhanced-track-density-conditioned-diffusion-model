"""Main Trans-TCDM training loop."""

import os

from typing import Dict
from importlib import import_module
import argparse
import json
import numpy as np
from matplotlib import pyplot as plt
import torch
from time import time
from torch import nn
from tqdm import tqdm
from torch.utils.data import DataLoader
from diffusion import GaussianDiffusionTrainer
from model import UNet
from torch.utils.tensorboard import SummaryWriter


def train(modelConfig: Dict):
    """训练函数 - 使用V4 Transformer编码器模型"""
    TrainDataset = import_module(f"datasets.{modelConfig['dataset']}").TrainDataset
    gpus = modelConfig['gpus']
    torch.cuda.set_device(modelConfig['output_device'])

    device = torch.device(modelConfig["device"])
    print(f"使用设备: {device}")
    
    # 创建必要的目录
    result_dir = modelConfig["result_dir"]
    save_dir = modelConfig["save_dir"]
    log_dir = modelConfig.get("log_dir", os.path.join(result_dir, "logs"))
    os.makedirs(result_dir, exist_ok=True)
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    
    # 初始化TensorBoard
    writer = SummaryWriter(log_dir=log_dir)
    print(f"TensorBoard日志保存至: {log_dir}")
    
    # 数据集
    train_data_dir = modelConfig['train_data_dir']
    dataset_z = TrainDataset(train_data_dir)
    loader_z = DataLoader(
        dataset_z, batch_size=modelConfig["batch_size"], shuffle=True, drop_last=True)
    
    print(f'训练数据量: {len(dataset_z)}')

    # 使用V4模型 - Transformer编码器
    net_model = UNet(
        T=modelConfig["T"], 
        num_labels=10, 
        ch=modelConfig["channel"], 
        ch_mult=modelConfig["channel_mult"],
        num_res_blocks=modelConfig["num_res_blocks"], 
        dropout=modelConfig["dropout"]
    ).to(device)
    
    # 打印模型参数量
    total_params = sum(p.numel() for p in net_model.parameters())
    trainable_params = sum(p.numel() for p in net_model.parameters() if p.requires_grad)
    print(f"模型总参数量: {total_params:,}")
    print(f"可训练参数量: {trainable_params:,}")
    print(f"模型架构: V4 - Transformer编码器")

    net_model = nn.DataParallel(net_model.cuda(), device_ids=gpus, output_device=gpus[0])
    net_model.train()
    start_epoch = 0
    
    optimizer = torch.optim.AdamW(
        net_model.parameters(), lr=modelConfig["lr"], weight_decay=1e-4)
    
    if modelConfig["training_load_weight"] is not None:
        pre_train = torch.load(
            os.path.join(save_dir, modelConfig["training_load_weight"]), 
            map_location=device)
        net_model.load_state_dict(pre_train['model'], strict=False)
        start_epoch += pre_train['epoch']
        start_epoch += 1
        print(f"加载预训练模型: {modelConfig['training_load_weight']}, epoch: {pre_train['epoch']}")

    trainer = GaussianDiffusionTrainer(
        net_model, modelConfig["beta_1"], modelConfig["beta_T"], modelConfig["T"]).to(device)

    # 学习率调度器 - 使用warmup + cosine decay
    def lam(epoch):
        warmup_epochs = 5
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        else:
            progress = (epoch - warmup_epochs) / (modelConfig["epoch"] - warmup_epochs)
            return 0.5 * (1 + np.cos(np.pi * progress))
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lam)

    # 开始训练
    best_loss = float('inf')
    best_model = {}
    loss_curve = []
    
    net_model.train()
    global_step = 0
    
    for e in range(start_epoch, modelConfig["epoch"] + start_epoch):
        loss_value = 0
        ii = 0
        epoch_start_time = time()

        with tqdm(loader_z, dynamic_ncols=False) as tqdmDataLoader:
            tqdmDataLoader.set_description(f'V4训练-epoch:{e}/{modelConfig["epoch"]+start_epoch}')
            for index, (labels, images, fibre_density) in enumerate(tqdmDataLoader):
                ii += 1
                global_step += 1

                optimizer.zero_grad()
                x_0 = images.clamp(0, 1).to(device)
                labels = labels.clamp(0, 1).to(device)
                fibre_density = fibre_density.to(device)

                loss = trainer(x_0, labels, fibre_density)
                loss.backward()
                loss_value += loss.item()
                loss_curve.append(loss.item())
                
                # 记录到TensorBoard
                writer.add_scalar('Loss/step', loss.item(), global_step)
                
                torch.nn.utils.clip_grad_norm_(net_model.parameters(), modelConfig["grad_clip"])
                optimizer.step()
                
                tqdmDataLoader.set_postfix(ordered_dict={
                    "loss": loss.item(),
                    "LR": optimizer.state_dict()['param_groups'][0]["lr"],
                    "img_shape": x_0.shape,
                })

        # 计算平均损失
        epoch_time = time() - epoch_start_time
        loss_value /= ii
        current_lr = optimizer.state_dict()['param_groups'][0]["lr"]
        
        # 记录epoch指标
        writer.add_scalar('Loss/epoch', loss_value, e)
        writer.add_scalar('LearningRate', current_lr, e)
        writer.add_scalar('Time/epoch', epoch_time, e)
        
        print(f'Epoch {e} | 平均损失: {loss_value:.6f} | 时间: {epoch_time:.2f}s | LR: {current_lr:.2e}')
        scheduler.step()

        if loss_value < best_loss:
            best_loss = loss_value
            best_model['model'] = net_model.state_dict()
            best_model['epoch'] = e

        # 保存检查点
        if (e + 1) % 1 == 0:
            content = {
                'model': net_model.state_dict(),
                'optim': optimizer.state_dict(),
                'epoch': e
            }
            torch.save(content, os.path.join(save_dir, f'ckpt_{e:0>4d}.pth'))

    # 保存最佳模型
    net_model.eval()
    torch.save(best_model, os.path.join(save_dir, 'best.pth'))
    print(f'最佳轮数: {best_model["epoch"]}, 最佳损失: {best_loss:.6f}')
    
    # 关闭TensorBoard
    writer.close()
    
    # 绘制损失曲线
    plt.figure(figsize=(10, 6))
    plt.plot(loss_curve)
    plt.title('Training Loss Curve (Trans-TCDM)')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.savefig(os.path.join(result_dir, 'loss_curve.png'))
    plt.close()
    print(f'损失曲线已保存至 {os.path.join(result_dir, "loss_curve.png")}')


def main():
    parser = argparse.ArgumentParser(description="Train Trans-TCDM on one dataset.")
    parser.add_argument("--config", required=True, help="Path to a dataset JSON config.")
    parser.add_argument("--gpu", type=int, default=0, help="CUDA device index.")
    parser.add_argument("--resume", default=None, help="Checkpoint filename in save_dir; reloads model weights only.")
    args = parser.parse_args()
    with open(args.config, encoding="utf-8") as f:
        config = json.load(f)
    config["gpus"] = [args.gpu]
    config["output_device"] = f"cuda:{args.gpu}"
    config["device"] = f"cuda:{args.gpu}"
    config["training_load_weight"] = args.resume
    torch.manual_seed(123)
    torch.cuda.manual_seed(123)
    torch.cuda.manual_seed_all(123)
    train(config)


if __name__ == "__main__":
    main()
