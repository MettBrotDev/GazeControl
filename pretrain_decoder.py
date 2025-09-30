import os
import time
import datetime
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms
from torch.utils.tensorboard import SummaryWriter
import torchvision.utils as vutils

from models import PretrainAutoencoder, ImageEncoderForPretrain, PerceptualLoss
from config import Config


def main():
    parser = argparse.ArgumentParser(description="Pretrain decoder as autoencoder")
    parser.add_argument("--config_module", default="config", help="config module name (e.g. 'config' or 'configL')")
    parser.add_argument("--single", action="store_true", help="Overfit a single random image sampled from the selected dataset")
    parser.add_argument("--steps", type=int, default=None, help="Override Config.PRETRAIN_STEPS")
    parser.add_argument("--batch-size", type=int, default=None, help="Override Config.PRETRAIN_BATCH_SIZE for pretraining")
    args = parser.parse_args()

    # Dynamically load config if requested
    global Config
    if args.config_module and args.config_module != "config":
        try:
            import importlib
            mod = importlib.import_module(args.config_module)
            Config = mod.Config
            print(f"Loaded config from {args.config_module}")
        except Exception as e:
            print(f"Failed to load config module '{args.config_module}', falling back to default. Error: {e}")

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(
        "runs",
        f"pretrain_decoder_{Config.DATA_SOURCE}_ch{getattr(Config,'DECODER_LATENT_CH',0)}_{timestamp}"
    )
    os.makedirs(run_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=run_dir)

    # Pretraining: train on [-1,1] with Tanh output
    transform_img = transforms.Compose([
        transforms.Resize(Config.IMG_SIZE),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x * 2.0 - 1.0),
    ])

    if Config.DATA_SOURCE == "mnist":
        transform_mnist = transforms.Compose([
            transforms.Resize(Config.IMG_SIZE),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.repeat(3,1,1)),
            transforms.Lambda(lambda x: x * 2.0 - 1.0),
        ])
        dataset = datasets.MNIST(root=Config.MNIST_DATA_DIR, train=True, transform=transform_mnist, download=True)
    elif Config.DATA_SOURCE == "cifar100":
        dataset = datasets.CIFAR100(root=Config.CIFAR100_DATA_DIR, train=True, transform=transform_img, download=True)
    else:
        dataset = datasets.CIFAR100(root=Config.CIFAR100_DATA_DIR, train=True, transform=transform_img, download=True)
    if args.single:
        # sample a single random item from dataset
        idx = torch.randint(0, len(dataset), (1,)).item()
        img_t, _ = dataset[idx]
        if img_t.dim() == 3:
            img_t = img_t.unsqueeze(0)
        dataset = TensorDataset(img_t, torch.zeros(1, dtype=torch.long))
        batch_size = 1
        shuffle = False
        print(f"Pretraining on a single random dataset image (idx={idx})")
    else:
        cfg_bs = max(1, int(getattr(Config, "PRETRAIN_BATCH_SIZE", 32)))
        batch_size = int(args.batch_size) if args.batch_size is not None else cfg_bs
        batch_size = max(1, batch_size)
        shuffle = True
    print(f"Using pretrain batch size: {batch_size}")

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0)

    # Use state_size matching main model hidden size for decoder compatibility
    enc_dim = int(Config.HIDDEN_SIZE)
    print(f"Pretrain AE: state_size={enc_dim}, dec_latent_ch={Config.DECODER_LATENT_CH}")

    autoenc = PretrainAutoencoder(state_size=enc_dim,
                                  img_size=Config.IMG_SIZE,
                                  decoder_latent_ch=Config.DECODER_LATENT_CH,
                                  out_activation="tanh").to(Config.DEVICE)

    # Optimizer + scheduler + loss (L1-dominant)
    lr = getattr(Config, "PRETRAIN_LR", 3e-3)
    opt = torch.optim.Adam(autoenc.parameters(), lr=lr, betas=(0.9, 0.99))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1000, Config.PRETRAIN_STEPS))
    criterion = PerceptualLoss().to(Config.DEVICE)
    l1_weight = float(getattr(Config, "PRETRAIN_L1_MIX", 5.0))
    l1 = nn.L1Loss(reduction="mean")

    def total_variation(x):
        tv_h = (x[:, :, 1:, :] - x[:, :, :-1, :]).abs().mean()
        tv_w = (x[:, :, :, 1:] - x[:, :, :, :-1]).abs().mean()
        return tv_h + tv_w
    use_amp = bool(getattr(Config, "PRETRAIN_USE_AMP", True))
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)

    steps = 0
    max_steps = args.steps if args.steps is not None else Config.PRETRAIN_STEPS
    autoenc.train()
    best_loss = float("inf")
    best_path = os.path.join(run_dir, "pretrained_decoder_best.pth")
    last_log_t = time.time()
    while steps < max_steps:
        for images, _ in loader:
            images = images.to(Config.DEVICE)
            pred = autoenc(images)
            loss_perc = criterion(pred, images)
            l1_loss = l1(pred, images)
            loss = loss_perc + l1_weight * l1_loss
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            steps += 1
            if steps % 50 == 0:
                writer.add_scalar("PretrainAE/loss", loss.item(), steps)
                writer.add_scalar("PretrainAE/l1_loss", l1_loss.item(), steps)
                writer.add_scalar("PretrainAE/loss_perc", loss_perc.item(), steps)
                # log last sample only, like main training (orig vs recon)
                with torch.no_grad():
                    img_last = (images[-1:].detach() + 1.0) / 2.0
                    pred_last = (pred[-1:].detach().clamp(-1, 1) + 1.0) / 2.0
                    grid = vutils.make_grid(torch.cat([img_last, pred_last], dim=0), nrow=2, normalize=True)
                writer.add_image("PretrainAE/original_vs_reconstruction", grid, steps)
                now = time.time()
                img_per_s = (50 * images.size(0)) / max(1e-6, (now - last_log_t))
                last_log_t = now
                print(f"step {steps:6d} | loss {loss.item():.4f} | lr {sched.get_last_lr()[0]:.2e} | {img_per_s:.1f} img/s")
            # Track best checkpoint
            if loss.item() < best_loss:
                best_loss = loss.item()
                torch.save(autoenc.decoder.state_dict(), best_path)
            if steps >= max_steps:
                break

    dec_path = os.path.join(run_dir, "pretrained_decoder_last.pth")
    torch.save(autoenc.decoder.state_dict(), dec_path)
    print(f"Saved pretrained decoder (last) to {dec_path}")
    if os.path.exists(best_path):
        print(f"Best checkpoint saved to {best_path} (loss={best_loss:.4f})")
        # Also export best to configured pretrained path for easy loading in train.py
        target = getattr(Config, "PRETRAINED_DECODER_PATH", "pretrained_components/pretrained_decoder.pth")
        os.makedirs(os.path.dirname(target), exist_ok=True)
        torch.save(autoenc.decoder.state_dict(), target)
        print(f"Exported best decoder to {target}")
    writer.close()


if __name__ == "__main__":
    main()
