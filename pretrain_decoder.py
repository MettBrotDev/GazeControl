import os
import time
import datetime
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, ConcatDataset, Dataset
from torchvision import datasets, transforms
import glob
from PIL import Image
# TensorBoard removed; using optional Weights & Biases instead
import torchvision.utils as vutils
try:
    import wandb
except Exception:
    wandb = None

from models import PretrainAutoencoder, ImageEncoderForPretrain, PerceptualLoss
from config import Config


def main():
    parser = argparse.ArgumentParser(description="Pretrain decoder as autoencoder")
    parser.add_argument("--config_module", default="config", help="config module name (e.g. 'config' or 'configL')")
    parser.add_argument("--single", action="store_true", help="Overfit a single random image sampled from the selected dataset")
    parser.add_argument("--steps", type=int, default=None, help="Override Config.PRETRAIN_STEPS")
    parser.add_argument("--batch-size", type=int, default=None, help="Override Config.PRETRAIN_BATCH_SIZE for pretraining")
    parser.add_argument("--no-perc", action="store_true", help="Disable perceptual loss during pretraining to save VRAM")
    parser.add_argument("--perc-resize", type=int, default=None, help="Resize to this square size before perceptual loss (e.g., 224)")
    parser.add_argument("--perceptual-weight", type=float, default=None, help="Scale factor for perceptual loss; overrides Config.PRETRAIN_PERC_WEIGHT")
    parser.add_argument("--wandb", action="store_true", help="Enable Weights & Biases logging")
    parser.add_argument("--wandb-project", type=str, default="GazeControl", help="W&B project name")
    parser.add_argument("--wandb-entity", type=str, default=None, help="W&B entity (team) name")
    parser.add_argument("--wandb-run-name", type=str, default=None, help="W&B run name")
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
    # TensorBoard disabled; use W&B if enabled
    wb = None
    if args.wandb and wandb is not None:
        # Attempt explicit login using environment API key to avoid no-tty prompts on clusters
        try:
            _api_key = os.environ.get("WANDB_API_KEY", None)
            if _api_key:
                try:
                    wandb.login(key=_api_key)
                    print("W&B: logged in via WANDB_API_KEY from environment")
                except Exception as e:
                    print(f"W&B: login via env key failed: {e}")
            else:
                # If no API key and not offline, W&B may attempt an interactive prompt and fail on clusters
                if os.environ.get("WANDB_MODE", "").lower() not in {"offline", "disabled", "dryrun"}:
                    print("W&B: WANDB_API_KEY is not set and WANDB_MODE is not offline; if this is a non-interactive job, set WANDB_MODE=offline or export WANDB_API_KEY.")
        except Exception:
            pass
        wb = wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=args.wandb_run_name or f"pretrain_decoder_{Config.DATA_SOURCE}_{timestamp}",
            config={
                "img_size": Config.IMG_SIZE,
                "batch_size": int(args.batch_size) if args.batch_size is not None else int(getattr(Config, "PRETRAIN_BATCH_SIZE", 32)),
                "lr": float(getattr(Config, "PRETRAIN_LR", 3e-3)),
                "perc": not args.no_perc,
                "perc_resize": args.perc_resize,
                "perc_weight": (float(getattr(args, "perceptual_weight", None))
                                  if getattr(args, "perceptual_weight", None) is not None
                                  else float(getattr(Config, "PRETRAIN_PERC_WEIGHT", 0.0))),
                "device": Config.DEVICE,
                "data_source": Config.DATA_SOURCE,
            },
            dir=run_dir,
        )

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
        # Load from local directories (e.g., Pathfinder) using same helper approach as train.py
        local_dirs = []
        if hasattr(Config, 'get_local_data_dirs'):
            try:
                local_dirs = list(Config.get_local_data_dirs())
            except Exception:
                pass
        if not local_dirs:
            ld = getattr(Config, 'LOCAL_DATA_DIR', None)
            if isinstance(ld, str):
                local_dirs = [ld]
            elif isinstance(ld, (list, tuple)):
                local_dirs = list(ld)

        class _SimpleImageDataset(Dataset):
            def __init__(self, data_dir, transform=None):
                self.transform = transform
                self.image_paths = []
                exts = ['*.jpg','*.jpeg','*.png','*.bmp','*.tiff','*.webp']
                for ext in exts:
                    self.image_paths.extend(glob.glob(os.path.join(data_dir, '**', ext), recursive=True))
                    self.image_paths.extend(glob.glob(os.path.join(data_dir, '**', ext.upper()), recursive=True))
                self.image_paths = sorted(list(set(self.image_paths)))
                # Skip expensive verification; handle bad files at load time
                print(f"Pretrain: found {len(self.image_paths)} images (unverified) in {data_dir}", flush=True)
            def __len__(self):
                return len(self.image_paths)
            def __getitem__(self, idx):
                p = self.image_paths[idx]
                try:
                    img = Image.open(p).convert('RGB')
                except Exception:
                    img = Image.new('RGB', (Config.IMG_SIZE[1], Config.IMG_SIZE[0]), color='black')
                if transform_img:
                    img = transform_img(img)
                return img, 0

        local_sets = []
        for d in local_dirs:
            if os.path.exists(d):
                ds = _SimpleImageDataset(d)
                if len(ds) > 0:
                    local_sets.append(ds)
                else:
                    print(f"Pretrain: directory {d} has no valid images; skipping")
            else:
                print(f"Pretrain: directory {d} does not exist; skipping")
        if not local_sets:
            raise ValueError(f"No valid local data directories for pretraining: {local_dirs}")
        dataset = ConcatDataset(local_sets) if len(local_sets) > 1 else local_sets[0]
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
    criterion = None if args.no_perc else PerceptualLoss().to(Config.DEVICE)
    # Backward-compat: accept either PRETRAIN_L1_WEIGHT (preferred) or fallback to PRETRAIN_L1_MIX
    l1_weight = float(getattr(Config, "PRETRAIN_L1_WEIGHT", getattr(Config, "PRETRAIN_L1_MIX", 5.0)))
    _perc_cli = getattr(args, "perceptual_weight", None)
    perc_weight = float(_perc_cli) if _perc_cli is not None else float(getattr(Config, "PRETRAIN_PERC_WEIGHT", 0.0))
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
            with torch.amp.autocast('cuda', enabled=use_amp):
                pred = autoenc(images)
                # Perceptual loss: optional and optionally on downscaled copies
                if criterion is not None:
                    if args.perc_resize is not None and args.perc_resize > 0:
                        sz = int(args.perc_resize)
                        imgs_p = F.interpolate(images, size=(sz, sz), mode='bilinear', align_corners=False)
                        preds_p = F.interpolate(pred, size=(sz, sz), mode='bilinear', align_corners=False)
                    else:
                        imgs_p, preds_p = images, pred
                    loss_perc = criterion(preds_p, imgs_p)
                else:
                    loss_perc = 0.0
                l1_loss = l1(pred, images)
                # Scale perceptual component by perc_weight
                if isinstance(loss_perc, torch.Tensor):
                    loss = perc_weight * loss_perc + l1_weight * l1_loss
                else:
                    loss = l1_weight * l1_loss
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            steps += 1
            if steps % 50 == 0:
                if wb is not None:
                    wandb.log({
                        "pretrain/loss": float(loss.item()),
                        "pretrain/l1": float(l1_loss.item()),
                        "pretrain/perc": float(loss_perc if not isinstance(loss_perc, torch.Tensor) else loss_perc.item()),
                        "pretrain/perc_weight": perc_weight,
                        "pretrain/l1_weight": l1_weight,
                        "lr": float(sched.get_last_lr()[0]),
                        "images_seen": steps * images.size(0),
                    }, step=steps)
                # log last sample only, like main training (orig vs recon)
                with torch.no_grad():
                    img_last = (images[-1:].detach() + 1.0) / 2.0
                    pred_last = (pred[-1:].detach().clamp(-1, 1) + 1.0) / 2.0
                    grid = vutils.make_grid(torch.cat([img_last, pred_last], dim=0), nrow=2, normalize=True)
                if wb is not None:
                    wandb.log({
                        "pretrain/orig_vs_recon": [wandb.Image(grid, caption=f"step {steps}")]
                    }, step=steps)
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
        if wb is not None:
            art = wandb.Artifact("pretrained_decoder", type="model")
            art.add_file(target)
            wandb.log_artifact(art)
    if wb is not None:
        wandb.finish()


if __name__ == "__main__":
    main()
