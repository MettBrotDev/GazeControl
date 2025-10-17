import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torchvision import models


def create_spatial_position_encoding(gaze_coords: torch.Tensor, encoding_dim: int) -> torch.Tensor:
    """Sinusoidal positional encoding of (x,y) gaze coords in [0,1].
    gaze_coords: (B,2)
    returns: (B,encoding_dim)
    """
    B = gaze_coords.shape[0]
    device = gaze_coords.device
    half = encoding_dim // 2
    # frequencies
    div_term = torch.exp(
        torch.arange(0, half, 2, device=device, dtype=torch.float) * (-(math.log(10000.0) / max(1, half)))
    )
    x = gaze_coords[:, 0:1]
    y = gaze_coords[:, 1:2]

    # allocate
    x_enc = torch.zeros(B, half, device=device)
    y_enc = torch.zeros(B, half, device=device)

    x_enc[:, 0::2] = torch.sin(x * div_term)
    x_enc[:, 1::2] = torch.cos(x * div_term)
    y_enc[:, 0::2] = torch.sin(y * div_term)
    y_enc[:, 1::2] = torch.cos(y * div_term)

    return torch.cat([x_enc, y_enc], dim=1)


class Encoder(nn.Module):
    """
    Simple encoder network that takes an image patch and returns a feature vector.
    Input: (B, 3, H, W)
    Output: (B, hidden_size)
    """
    def __init__(self, hidden_size=256, input_size=(16, 16), c1: int = 24, c2: int = 48):
        super().__init__()
        self.input_size = input_size  # (H, W)
        self.c1 = int(c1)
        self.c2 = int(c2)
        # Input-agnostic encoder: keep resolution with stride=1, then adaptively pool to 2x2
        self.encoder = nn.Sequential(
            nn.Conv2d(3, self.c1, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.c1, self.c2, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
        )
        self.spatial_pool = nn.AdaptiveAvgPool2d((2, 2))

    def forward(self, x):
        x = self.encoder(x)
        x = self.spatial_pool(x)
        x = x.view(x.size(0), -1)  # Flatten (B, c2*2*2)
        return x


class DecoderCNN(nn.Module):
    """Simple, modest-capacity decoder for up to ~300x300 outputs.
    Uses bilinear upsample + 3x3 conv blocks to 64x64, then bilinear-resizes to the final size.
    Channel widths scale with latent_ch but remain small. Output is always tanh in [-1,1].
    """
    def __init__(self, state_size: int, out_size=(32, 32), latent_ch: int = 96):
        super().__init__()
        self.out_h, self.out_w = out_size

        c1 = int(latent_ch)
        c2 = max(16, c1 // 2)
        c3 = max(16, c2 // 2)
        c4 = max(8, c3 // 2)

        # Project latent to 4x4 feature map
        self.fc = nn.Linear(state_size, 128 * 4 * 4)
        self.unflatten = nn.Unflatten(1, (128, 4, 4))

        # 4x4 -> 8x8
        self.up1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.conv1 = nn.Conv2d(128, c1, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(c1) if c1 >= 16 else nn.GroupNorm(max(1, min(4, c1)), c1)
        self.act1 = nn.LeakyReLU(inplace=True)

        # 8x8 -> 16x16
        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.conv2 = nn.Conv2d(c1, c2, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(c2) if c2 >= 16 else nn.GroupNorm(max(1, min(4, c2)), c2)
        self.act2 = nn.LeakyReLU(inplace=True)

        # 16x16 -> 32x32
        self.up3 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.conv3 = nn.Conv2d(c2, c3, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(c3) if c3 >= 16 else nn.GroupNorm(max(1, min(4, c3)), c3)
        self.act3 = nn.LeakyReLU(inplace=True)

        # 32x32 -> 64x64
        self.up4 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.conv4 = nn.Conv2d(c3, c4, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(c4) if c4 >= 16 else nn.GroupNorm(max(1, min(4, c4)), c4)
        self.act4 = nn.LeakyReLU(inplace=True)

    # final RGB conv (64x64)
        self.conv_out = nn.Conv2d(c4, 3, kernel_size=3, padding=1)
        
        # Initialize output bias to positive value to prevent black collapse
        # tanh(0.5) ≈ 0.46, which maps to ~0.73 in [0,1] space (gray-ish start)
        nn.init.constant_(self.conv_out.bias, 0.5)


    def forward(self, h):
        # Latent to 4x4
        x = self.fc(h)
        x = self.unflatten(x)
        # Upsample + conv blocks
        x = self.up1(x); x = self.conv1(x); x = self.bn1(x); x = self.act1(x)
        x = self.up2(x); x = self.conv2(x); x = self.bn2(x); x = self.act2(x)
        x = self.up3(x); x = self.conv3(x); x = self.bn3(x); x = self.act3(x)
        x = self.up4(x); x = self.conv4(x); x = self.bn4(x); x = self.act4(x)
        x = self.conv_out(x)
        x = torch.tanh(x)
        if x.shape[-2] != self.out_h or x.shape[-1] != self.out_w:
            x = F.interpolate(x, size=(self.out_h, self.out_w), mode='bilinear', align_corners=False)
        return x


class FusionMLP(nn.Module):
    """A small MLP to fuse what (multi-scale features) and where (pos enc + raw gaze).
    This mimics the Glimpse Network from the RAM paper.
    """
    def __init__(self, in_dim: int, out_dim: int, hidden_mul: float = 2.0):
        super().__init__()
        hidden = max(out_dim, int(hidden_mul * out_dim))
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.LayerNorm(hidden),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x):
        return self.net(x)


class GazeControlModel(nn.Module):
    """
    Basic model with shared CNN encoder, plain LSTM memory, and CNN decoder to full image.
    Uses Multi-scale glimpses similar to the Glimpse sensor from the RAM paper.
    - Input each step: one patch (B,3,Hc,Wc) or list/tuple of k patches for multi-scale.
    - Memory: nn.LSTM over steps.
    - Output each step: reconstructed full image (B,3,H,W).
    """
    def __init__(self, encoder_output_size, state_size=768, img_size=(32, 32), fovea_size=(24, 24), pos_encoding_dim: int = 32, lstm_layers: int = 2, decoder_latent_ch: int = 48, k_scales: int = 3, fuse_to_dim: int | None = None, fusion_hidden_mul: float = 2.0, encoder_c1: int | None = None, encoder_c2: int | None = None):
        super().__init__()
        # Allow scaling encoder channel widths via optional args
        enc_c1 = 24 if encoder_c1 is None else int(encoder_c1)
        enc_c2 = 48 if encoder_c2 is None else int(encoder_c2)
        self.encoder = Encoder(hidden_size=encoder_output_size, input_size=fovea_size, c1=enc_c1, c2=enc_c2)
        self.state_size = state_size
        self.img_size = img_size
        self.pos_dim = pos_encoding_dim
        self.k_scales = max(1, int(k_scales))

        # LSTM input dimension is the fusion output. If fuse_to_dim is None, keep at k*E + pos + 2
        fused_dim = fuse_to_dim if fuse_to_dim is not None else (encoder_output_size * self.k_scales + self.pos_dim + 2)
        fusion_in = encoder_output_size * self.k_scales + self.pos_dim + 2
        self.fusion = FusionMLP(in_dim=fusion_in, out_dim=fused_dim, hidden_mul=fusion_hidden_mul)
        self.input_dim = fused_dim

        # LSTM memory
        self.lstm = nn.LSTM(input_size=self.input_dim, hidden_size=state_size, num_layers=lstm_layers, batch_first=True)
        self.lstm_layers = lstm_layers

        # CNN decoder from hidden state to image (always tanh for [-1,1] images)
        self.decoder = DecoderCNN(state_size=state_size, out_size=img_size, latent_ch=decoder_latent_ch)

    def init_memory(self, batch_size, device):
        """Return zero (h0, c0) for LSTM of shape (num_layers, B, state_size)."""
        h0 = torch.zeros(self.lstm_layers, batch_size, self.state_size, device=device)
        c0 = torch.zeros(self.lstm_layers, batch_size, self.state_size, device=device)
        return (h0, c0)

    def decode_from_state(self, lstm_state):
        """Decode final reconstruction only from the current LSTM hidden state (top layer)."""
        h = lstm_state[0][-1]  # (B,H) from top layer
        return self.decoder(h)

    def forward(self, patches, lstm_state, gaze):
        """
        Args:
            patches: list/tuple of length k with each tensor shaped (B,3,Hc,Wc)
            lstm_state: (h,c) where each is (num_layers,B,state_size)
            gaze: (B,2) normalized [0,1]
        Returns:
            reconstruction: (B,3,H,W)
            next_state: next (h,c)
        """
        # Expect a sequence of k patches; concatenate encoded features across scales
        feats = [self.encoder(p) for p in patches]
        feat = torch.cat(feats, dim=1)
        pos_enc = create_spatial_position_encoding(gaze, self.pos_dim)  # (B,pos_dim)
        fused_in = torch.cat([feat, pos_enc, gaze], dim=1)  # (B, k*E + pos + 2)
        fused = self.fusion(fused_in)                  # (B, fused_dim)
        step_in = fused.unsqueeze(1)                   # (B,1,input_dim)
        lstm_out, next_state = self.lstm(step_in, lstm_state)  # lstm_out: (B,1,H)
        h = lstm_out.squeeze(1)                    # (B,H)
        reconstruction = self.decoder(h)
        return reconstruction, next_state


class Agent(nn.Module):
    """Actor-Critic head that consumes LSTM hidden state and gaze context.
    """
    def __init__(self, state_size: int, pos_encoding_dim: int, num_actions: int = 8):
        super().__init__()
        self.pos_dim = pos_encoding_dim
        rl_in_dim = state_size + self.pos_dim + 2
        shared_hidden = max(128, state_size // 2)
        head_hidden = max(64, state_size // 4)

        # Shared representation over [h, pos-enc, gaze]
        self.shared = nn.Sequential(
            nn.Linear(rl_in_dim, shared_hidden),
            nn.GELU(),
            nn.LayerNorm(shared_hidden),
        )

        # Policy and value heads operate on shared features
        self.policy_head = nn.Sequential(
            nn.Linear(shared_hidden, head_hidden),
            nn.ReLU(),
            nn.Linear(head_hidden, num_actions),
        )
        self.value_head = nn.Sequential(
            nn.Linear(shared_hidden, head_hidden),
            nn.ReLU(),
            nn.Linear(head_hidden, 1),
        )

    def policy_value(self, h: torch.Tensor, gaze: torch.Tensor):
        pe = create_spatial_position_encoding(gaze, self.pos_dim)
        rl_in = torch.cat([h, pe, gaze], dim=1)
        shared = self.shared(rl_in)
        logits = self.policy_head(shared)
        value = self.value_head(shared).squeeze(-1)
        return logits, value


class ImageEncoderForPretrain(nn.Module):
    """Slightly stronger AE encoder: 3 strided downsamples + 1 extra 3x3 conv.
    Keeps compute modest while capturing thin structures better.
    """
    def __init__(self, state_size: int, img_size=(32, 32)):
        super().__init__()
        c1, c2, c3 = 96, 192, 128
        self.features = nn.Sequential(
            nn.Conv2d(3, c1, 3, stride=2, padding=1), nn.BatchNorm2d(c1), nn.GELU(),
            nn.Conv2d(c1, c2, 3, stride=2, padding=1), nn.BatchNorm2d(c2), nn.GELU(),
            nn.Conv2d(c2, c3, 3, stride=2, padding=1), nn.BatchNorm2d(c3), nn.GELU(),
            nn.Conv2d(c3, c3, 3, stride=1, padding=1), nn.BatchNorm2d(c3), nn.GELU(),
        )
        # Fixed spatial size before FC for arbitrary input sizes
        self.adapt_pool = nn.AdaptiveAvgPool2d((4, 4))
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(c3 * 4 * 4, state_size)

    def forward(self, x):
        x = self.features(x)
        x = self.adapt_pool(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x


class PretrainAutoencoder(nn.Module):
    """Autoencoder to pretrain the DecoderCNN using full-image reconstruction.
    The encoder projects to state_size so the decoder learns with the same input dimension
    as in the main model (HIDDEN_SIZE).
    """
    def __init__(self, state_size: int, img_size=(32, 32), decoder_latent_ch: int = 48):
        super().__init__()
        self.encoder = ImageEncoderForPretrain(state_size=state_size, img_size=img_size)
        self.decoder = DecoderCNN(state_size=state_size, out_size=img_size, latent_ch=decoder_latent_ch)

    def forward(self, x):
        h = self.encoder(x)
        out = self.decoder(h)
        return out
    


class PerceptualLoss(nn.Module):
    """Perceptual loss using VGG19 features, with ImageNet normalization."""
    def __init__(self, layer: str = 'features.16'):
        super().__init__()
        vgg = models.vgg19(weights=models.VGG19_Weights.DEFAULT).features
        self.vgg_slice = nn.Sequential(*list(vgg.children())[:17])
        self.vgg_slice.eval()  # use eval mode for stable features
        for p in self.vgg_slice.parameters():
            p.requires_grad = False
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))


    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        # Inputs expected in [-1,1]; convert to [0,1] then normalize for VGG
        mean = self.mean.to(device=x.device, dtype=self.mean.dtype)
        std = self.std.to(device=y.device, dtype=self.std.dtype)
        x01 = (x.clamp(-1.0, 1.0) + 1.0) * 0.5
        y01 = (y.clamp(-1.0, 1.0) + 1.0) * 0.5
        x_in = (x01 - mean) / std
        y_in = (y01 - mean) / std
        # Allow outer autocast to control precision for reduced VRAM
        x_vgg = self.vgg_slice(x_in)
        y_vgg = self.vgg_slice(y_in)
        loss = nn.functional.mse_loss(x_vgg, y_vgg)
        return loss