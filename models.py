import torch
import torch.nn as nn
import math


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
    def __init__(self, hidden_size=256, input_size=(16, 16)):
        super().__init__()
        self.input_size = input_size  # (H, W)
        # Simple 3-layer CNN for 16x16 input
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 12, 4, stride=2, padding=1),            # [batch, 12, 8, 8] for 16x16 input
            nn.ReLU(),
            nn.Conv2d(12, 24, 4, stride=2, padding=1),           # [batch, 24, 4, 4]
            nn.ReLU(),
            nn.Conv2d(24, 48, 4, stride=2, padding=1),           # [batch, 48, 2, 2]
            nn.ReLU(),
        )

    def forward(self, x):
        x = self.encoder(x)
        x = x.view(x.size(0), -1)  # Flatten spatial features to vector
        return x


class DecoderCNN(nn.Module):
    """Simple CNN decoder that takes flattened features and outputs an image.
    Uses latent_ch for the spatial feature channels so capacity can be tuned from config.
    """
    def __init__(self, state_size: int, out_size=(32, 32), latent_ch: int = 48, out_activation: str = "sigmoid"):
        super().__init__()
        self.out_h, self.out_w = out_size
        self.latent_ch = latent_ch

        # Project flattened features back to spatial features
        self.fc = nn.Linear(state_size, self.latent_ch * 4 * 4)

        c1 = self.latent_ch
        c2 = max(16, self.latent_ch // 2)
        c3 = max(8, c2 // 2)

        self.up1 = nn.ConvTranspose2d(c1, c2, 4, stride=2, padding=1)
        self.up2 = nn.ConvTranspose2d(c2, c3, 4, stride=2, padding=1)
        self.up3 = nn.ConvTranspose2d(c3, 3, 4, stride=2, padding=1)

        if out_activation == "tanh":
            self.output_act = nn.Tanh()
        else:
            self.output_act = nn.Sigmoid()

    def forward(self, h):
        """Decode from a flat state vector h -> (B,3,H,W)."""
        # h is flattened features: (B, state_size)
        B = h.size(0)
        x = self.fc(h).view(B, self.latent_ch, 4, 4)
        x = nn.functional.leaky_relu(self.up1(x), 0.1, inplace=True)
        x = nn.functional.leaky_relu(self.up2(x), 0.1, inplace=True)
        x = self.up3(x)
        x = self.output_act(x)
        return x


class GazeControlModel(nn.Module):
    """
    Basic model with CNN encoder, plain LSTM memory, and CNN decoder to full image.
    - Input each step: cropped patch around current gaze.
    - Memory: nn.LSTM over steps.
    - Output each step: reconstructed full image (B,3,H,W).
    """
    def __init__(self, encoder_output_size, state_size=768, img_size=(32, 32), fovea_size=(24, 24), pos_encoding_dim: int = 32, lstm_layers: int = 2, decoder_latent_ch: int = 48):
        super().__init__()
        self.encoder = Encoder(hidden_size=encoder_output_size, input_size=fovea_size)
        self.state_size = state_size
        self.img_size = img_size
        self.pos_dim = pos_encoding_dim
        # include raw gaze (2) + positional encoding
        self.input_dim = encoder_output_size + self.pos_dim + 2

        # LSTM memory
        self.lstm = nn.LSTM(input_size=self.input_dim, hidden_size=state_size, num_layers=lstm_layers, batch_first=True)
        self.lstm_layers = lstm_layers

        # CNN decoder from hidden state to image (default sigmoid for main training)
        self.decoder = DecoderCNN(state_size=state_size, out_size=img_size, latent_ch=decoder_latent_ch, out_activation="sigmoid")

        # RL heads (actor-critic) on top of hidden state h_t and gaze context (pos enc + raw coords)
        rl_in_dim = state_size + self.pos_dim + 2
        # Discrete policy with 8 actions (8 direction moves)
        self.policy_head = nn.Sequential(
            nn.Linear(rl_in_dim, state_size // 2), nn.ReLU(), nn.Linear(state_size // 2, 8)
        )
        self.value_head = nn.Sequential(
            nn.Linear(rl_in_dim, state_size // 2), nn.ReLU(), nn.Linear(state_size // 2, 1)
        )

    def init_memory(self, batch_size, device):
        """Return zero (h0, c0) for LSTM of shape (num_layers, B, state_size)."""
        h0 = torch.zeros(self.lstm_layers, batch_size, self.state_size, device=device)
        c0 = torch.zeros(self.lstm_layers, batch_size, self.state_size, device=device)
        return (h0, c0)

    def decode_from_state(self, lstm_state):
        """Decode final reconstruction only from the current LSTM hidden state (top layer)."""
        h = lstm_state[0][-1]  # (B,H) from top layer
        return self.decoder(h)

    def forward(self, patch, lstm_state, gaze):
        """
        Args:
            patch: (B,3,Hc,Wc) cropped image patch
            lstm_state: (h,c) where each is (num_layers,B,state_size)
            gaze: (B,2) normalized [0,1]
        Returns:
            reconstruction: (B,3,H,W)
            next_state: next (h,c)
        """
        B = patch.size(0)
        feat = self.encoder(patch)                 # (B, encoder_output_size)
        pe = create_spatial_position_encoding(gaze, self.pos_dim)  # (B,pos_dim)
        step_in = torch.cat([feat, pe, gaze], dim=1)  # add raw gaze
        step_in = step_in.unsqueeze(1)             # (B,1,input_dim)
        lstm_out, next_state = self.lstm(step_in, lstm_state)  # lstm_out: (B,1,H)
        h = lstm_out.squeeze(1)                    # (B,H)
        reconstruction = self.decoder(h)
        return reconstruction, next_state

    # ---- RL API ----
    def policy_value(self, h, gaze):
        """Return discrete-action logits (8-way) and value given hidden state h and current gaze.
        h: (B,H), gaze: (B,2)
        returns: logits (B,8), value (B,)
        """
        pe = create_spatial_position_encoding(gaze, self.pos_dim)  # (B,pos_dim)
        rl_in = torch.cat([h, pe, gaze], dim=1)
        logits = self.policy_head(rl_in)  # (B,8)
        value = self.value_head(rl_in).squeeze(-1)  # (B,)
        return logits, value


class ImageEncoderForPretrain(nn.Module):
    """Encodes a full image (3,H,W) to flattened features for decoder pretraining."""
    def __init__(self, state_size: int, img_size=(32, 32)):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 12, 4, stride=2, padding=1),            # [batch, 12, 16, 16]
            nn.ReLU(),
            nn.Conv2d(12, 24, 4, stride=2, padding=1),           # [batch, 24, 8, 8]
            nn.ReLU(),
            nn.Conv2d(24, 48, 4, stride=2, padding=1),           # [batch, 48, 4, 4]
            nn.ReLU(),
        )

    def forward(self, x):
        x = self.encoder(x)  # (B, 48, 4, 4)
        x = x.view(x.size(0), -1)  # Flatten: (B, 768)
        return x


class PretrainAutoencoder(nn.Module):
    """Autoencoder to pretrain the DecoderCNN using full-image reconstruction."""
    def __init__(self, state_size: int, img_size=(32, 32), decoder_latent_ch: int = 48, out_activation: str = "sigmoid"):
        super().__init__()
        self.encoder = ImageEncoderForPretrain(state_size=state_size, img_size=img_size)
        self.decoder = DecoderCNN(state_size=state_size, out_size=img_size, latent_ch=decoder_latent_ch, out_activation=out_activation)

    def forward(self, x):
        h = self.encoder(x)
        out = self.decoder(h)
        return out