import os
import copy
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.optim.lr_scheduler import OneCycleLR
from config import cfg
from dataset import CustomDataset
from Models import VQVAE, UNetModel, DiffusionTrainer, EMA
from utils import setup_global_seed, global_mean_std, filter_optimizer, save_loss
cfg['epochs'] = 500


def train_epoch(ema, model, ae, loss, train_loader, optimizer, scheduler, device, epoch, global_scale):
    model.train()
    avg_loss = 0
    with tqdm(train_loader, desc=f'epoch {epoch + 1}', dynamic_ncols=True) as tqdmTrainLoader:
        for y, real_s, init_s, seedvox in tqdmTrainLoader:
            real_s = real_s.to(device)
            optimizer.zero_grad()
            ze = ae.encode(real_s)
            ze = ze * global_scale
            pred_noise, noise = model(ze)
            pred_loss = loss(pred_noise, noise)
            pred_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            ema.update(model)
            avg_loss += pred_loss.item()

    return avg_loss / len(train_loader)


def eval_epoch(model, ae, loss, eval_loader, device, epoch, global_scale):
    model.eval()
    avg_loss = 0
    with tqdm(eval_loader, desc=f'epoch {epoch + 1}', dynamic_ncols=True) as tqdmEvalLoader:
        with torch.no_grad():
            for y, real_s, init_s, seedvox in tqdmEvalLoader:
                real_s = real_s.to(device)
                ze = ae.encode(real_s)
                ze = ze * global_scale
                pred_noise, noise = model(ze)
                pred_loss = loss(pred_noise, noise)
                avg_loss += pred_loss.item()
    return avg_loss / len(eval_loader)


if __name__ == "__main__":
    setup_global_seed()
    train_path = os.path.join("Datasets", "train.mat")
    val_path = os.path.join("Datasets", "val.mat")

    train_dataset = CustomDataset(train_path)
    val_dataset = CustomDataset(val_path)
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg['batch_size'],
        shuffle=True,
        num_workers=6,
        persistent_workers=True,
        pin_memory=True,
        drop_last=True)
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg['batch_size'],
        shuffle=False,
        num_workers=2,
        persistent_workers=True,
        pin_memory=True)

    best_loss = float('inf')
    best_weights = None
    loss = nn.MSELoss()

    ae = VQVAE(6004, 128)
    try:
        ae.load_state_dict(
            torch.load("./Weights/auto.pth",  map_location='cpu', weights_only=True))
    except Exception as e:
        print(f"[错误] 加载失败: {type(e).__name__}: {e}")
    ae = ae.to(cfg['device'])
    ae.eval()

    global_scale = global_mean_std(
        ae, train_loader, cfg['device'])
    model = DiffusionTrainer(
        UNetModel(1, 64, 1, 4, 1), cfg['diffusion']).to(cfg['device'])
    ema = EMA()
    ema.register(model)

    decay_params, no_decay_params = filter_optimizer(model)
    optimizer_params = [
        {
            'params': decay_params,
            'weight_decay': 0.01
        },
        {
            'params': no_decay_params,
            'weight_decay': 0.0
        }
    ]

    optimizer = torch.optim.AdamW(
        optimizer_params,
        lr=5e-4,
        betas=(0.9, 0.999),
    )

    scheduler = OneCycleLR(
        optimizer,
        max_lr=5e-4,                                # 初始学习率
        epochs=cfg['epochs'],                       # 总 Epoch 数
        steps_per_epoch=len(train_loader),
        pct_start=0.1,                              # 10% 的时间用于预热
        anneal_strategy='cos',                      # 余弦退火衰减
        div_factor=5e3,                             # 1e-7
        final_div_factor=1                          # 1e-7
    )

    for epoch in range(cfg['epochs']):
        train_loss_val = train_epoch(
            ema, model, ae, loss, train_loader, optimizer, scheduler, cfg['device'], epoch, global_scale)
        ema_model = copy.deepcopy(model).eval()
        ema.ema(ema_model)
        eval_loss_val = eval_epoch(
            ema_model, ae, loss, val_loader, cfg['device'], epoch, global_scale)

        if best_loss > eval_loss_val:
            best_loss = eval_loss_val
            best_weights = copy.deepcopy(model.state_dict())

        logFile = os.path.join("Logs", "train_df.txt")
        with open(logFile, 'a') as file:
            current_lr = optimizer.param_groups[0]['lr']
            file.write(f"Epoch {epoch + 1}: rate = {current_lr:.1e}\n")
            file.write(
                f'epoch:{epoch + 1} train_loss={train_loss_val :.4f}\n')
            file.write(
                f'epoch:{epoch + 1} val_loss={eval_loss_val :.4f}\n')

    checkpoint = {
        'global_scale': global_scale,
        'state_dict': best_weights
    }
    torch.save(checkpoint, 'Weights/diffusion.pth')
