import torch
import torch.nn as nn
import math

def create_spatial_position_encoding(gaze_coords, encoding_dim):
    """
    Create rich spatial position encoding for gaze coordinates.
    Args:
        gaze_coords: (B, 2) normalized coordinates [0,1]
        encoding_dim: dimension of encoding
    Returns:
        (B, encoding_dim) spatial encoding
    """
    batch_size = gaze_coords.shape[0] # We dont do batching bcs of rl so B should always be 1
    device = gaze_coords.device
    
    # Split encoding dimension between x and y
    half_dim = encoding_dim // 2
    
    # Create frequency basis
    div_term = torch.exp(torch.arange(0, half_dim, 2, device=device, dtype=torch.float) * 
                        -(math.log(10000.0) / half_dim))
    
    x_pos = gaze_coords[:, 0:1]  # (B, 1) We dont do batching bcs of rl so B is always 1
    y_pos = gaze_coords[:, 1:2]  # (B, 1)
    
    # Sinusoidal encoding
    x_encoding = torch.zeros(batch_size, half_dim, device=device)
    y_encoding = torch.zeros(batch_size, half_dim, device=device)
    
    x_encoding[:, 0::2] = torch.sin(x_pos * div_term)
    x_encoding[:, 1::2] = torch.cos(x_pos * div_term)
    y_encoding[:, 0::2] = torch.sin(y_pos * div_term)  
    y_encoding[:, 1::2] = torch.cos(y_pos * div_term)
    
    spatial_encoding = torch.cat([x_encoding, y_encoding], dim=1)
    
    return spatial_encoding

class Encoder(nn.Module):
    """
    Encoder network that takes a foveated image patch and returns a feature vector.
    Input: (B, 3, H, W) where H and W are multiples of 16
    Output: (B, hidden_size)
    """
    def __init__(self, hidden_size=512, input_size=(16, 16)):
        super().__init__()
        self.input_size = input_size  # (H, W)
        
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),  # HxW -> H/2xW/2
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1), # H/2xW/2 -> H/4xW/4
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1), # H/4xW/4 -> H/8xW/8
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1), # H/8xW/8 -> H/8xW/8 (no stride)
            nn.ReLU(),
        )

        # Calculate FC layer size based on input_size
        feature_h, feature_w = input_size[0] // 8, input_size[1] // 8
        fc_input_size = 128 * feature_h * feature_w
        self.fc = nn.Linear(fc_input_size, hidden_size)

    def forward(self, x):
        x = self.cnn(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

class SpatialMemoryCore(nn.Module):
    """
    Spatial memory core with learnable memory operations.
    The model learns HOW to write to memory, not just WHERE.
    Using a compact 2D memory with channel size `memory_dim`.
    """
    def __init__(self, encoder_feature_dim, memory_dim=4, memory_size=(32, 32), pos_encoding_dim=64, kernel_size=11):
        super().__init__()
        self.encoder_feature_dim = encoder_feature_dim
        self.memory_dim = memory_dim
        self.memory_size = memory_size
        self.pos_encoding_dim = pos_encoding_dim
        self.kernel_size = kernel_size  # odd 5/7/9
        # Fixed Gaussian width (in pixels), derived from kernel size
        self.sigma_pixels = float(kernel_size) / 3.0

        # Precompute neighborhood offsets as buffers (moved with device)
        r = kernel_size // 2
        offs = torch.arange(-r, r + 1, dtype=torch.long)
        ox, oy = torch.meshgrid(offs, offs, indexing='xy')
        self.register_buffer('ox', ox.reshape(-1), persistent=False)  # (K,)
        self.register_buffer('oy', oy.reshape(-1), persistent=False)  # (K,)

        # Input gate - processes encoder features + positional encoding
        self.input_gate = nn.Sequential(
            nn.Linear(encoder_feature_dim + pos_encoding_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU()
        )
        
        # Writer heads: produce K distinct neighbor values and a learned per-cell strength from (prev, candidate)
        K = kernel_size * kernel_size
        self.K = K
        self.write_value = nn.Linear(128, memory_dim * K)    # (B, K*M)
        self.strength_head = nn.Sequential(                  # (B,1) gate per neighbor cell
            nn.Linear(2 * memory_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
        # Output gate - consumes compact memory map
        self.output_gate = nn.Sequential(
            nn.Conv2d(memory_dim, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(3),  # Keep 3x3 spatial structure
            nn.Flatten(),
            nn.Linear(64 * 9, 64 * 9)  # 64 channels * 3*3 spatial = 576 features
        )

    def init_cell_state(self, batch_size, device):
        """Initialize empty spatial cell state (like LSTM cell state)."""
        return torch.zeros(batch_size, self.memory_dim, 
                          self.memory_size[0], self.memory_size[1], 
                          device=device)
    
    def update_cell_state(self, cell_state, features, gaze_coords):
        """Write different content into a kxk neighborhood using Gaussian geometry and learned per-cell gates (vectorized)."""
        batch_size = cell_state.shape[0]
        device = cell_state.device

        H, W = self.memory_size
        M = self.memory_dim
        k = self.kernel_size
        K = self.K
        r = k // 2
        
        # Position encoding and input gating
        pos_encoding = create_spatial_position_encoding(gaze_coords, self.pos_encoding_dim)
        gate_input = torch.cat([features, pos_encoding], dim=1)  # (B, encoder_feature_dim + pos)
        gate_output = self.input_gate(gate_input)                # (B, 128)
        
        # Predict neighbor-specific write vectors
        write_vals = self.write_value(gate_output).view(batch_size, K, M)  # (B,K,M)
        
        # Continuous gaze to pixel coords
        gx = torch.clamp(gaze_coords[:, 0] * (W - 1), 0.0, W - 1.0)  # (B,)
        gy = torch.clamp(gaze_coords[:, 1] * (H - 1), 0.0, H - 1.0)  # (B,)
        x0 = torch.floor(gx).long(); y0 = torch.floor(gy).long()
        
        # Neighbor coords per batch using cached offsets
        xs = (x0.unsqueeze(1) + self.ox.unsqueeze(0)).clamp(0, W - 1)  # (B,K)
        ys = (y0.unsqueeze(1) + self.oy.unsqueeze(0)).clamp(0, H - 1)  # (B,K)
        
        # Gaussian weights based on distance to continuous gaze with fixed sigma
        dx = (xs.float() - gx.unsqueeze(1))  # (B,K)
        dy = (ys.float() - gy.unsqueeze(1))  # (B,K)
        d2 = dx * dx + dy * dy               # (B,K)
        denom = (2.0 * (self.sigma_pixels ** 2))
        gauss = torch.exp(-d2 / max(1e-6, denom))       # (B,K)
        neigh_weights = gauss / gauss.sum(dim=1, keepdim=True).clamp_min(1e-8)  # (B,K)
        
        # Gather previous memory values for all neighbors from the original cell_state
        cell_flat = cell_state.view(batch_size, M, H * W)                 # (B,M,HW)
        idx_flat = (ys * W + xs).long()                                   # (B,K)
        prev_all = torch.gather(cell_flat, 2, idx_flat.unsqueeze(1).expand(batch_size, M, K))
        prev_all = prev_all.transpose(1, 2)                               # (B,K,M)
        
        # Compute learned gates for all neighbors in parallel
        gate_in = torch.cat([prev_all, write_vals], dim=-1)               # (B,K,2M)
        g_all = self.strength_head(gate_in.reshape(batch_size * K, 2 * M)).view(batch_size, K, 1)  # (B,K,1)
        
        # Combine geometry and learned gate
        w_all = neigh_weights.unsqueeze(-1) * g_all                       # (B,K,1)
        
        # Blend new values without read-after-write
        new_vals = (1.0 - w_all) * prev_all + w_all * write_vals          # (B,K,M)
        
        # Scatter-add weighted contributions back to spatial grid and accumulate weights
        idx_b1k = idx_flat.unsqueeze(1)                                   # (B,1,K)
        den = torch.zeros(batch_size, 1, H * W, device=device, dtype=cell_state.dtype)
        den.scatter_add_(2, idx_b1k, w_all.squeeze(-1).unsqueeze(1))      # (B,1,HW)
        
        num = torch.zeros(batch_size, M, H * W, device=device, dtype=cell_state.dtype)
        num.scatter_add_(2, idx_b1k.expand(batch_size, M, K), (w_all * new_vals).permute(0, 2, 1))  # (B,M,HW)
        
        prev_flat = cell_flat                                             # (B,M,HW)
        keep = (1.0 - den.clamp_max(1.0)).expand(-1, M, -1)               # (B,M,HW)
        updated_flat = keep * prev_flat + num                              # (B,M,HW)
        updated = updated_flat.view(batch_size, M, H, W)
        
        return updated
    
    def compute_output(self, cell_state):
        """Compute output from cell state using output gate (like LSTM output)."""
        return self.output_gate(cell_state)
    
    def forward(self, features, cell_state, gaze_coords):
        """
        Process current features and update cell state (like LSTM forward pass).
        Returns the memory readout (after output_gate) and the updated cell state.
        """
        # Update cell state using compact writer
        updated_cell_state = self.update_cell_state(cell_state, features, gaze_coords)
        
        # Compute output from updated cell state (memory readout for Agent)
        output = self.compute_output(updated_cell_state)  # (B, 576)
        
        return output, updated_cell_state

class Agent(nn.Module):
    """
    Policy network (Agent) that decides the next gaze movement and whether to stop.
    Outputs a tensor of shape (B, 3) representing (dx, dy, stop_logit).
    """
    def __init__(self, hidden_size=512):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(hidden_size, hidden_size//2),
            nn.ReLU(),
        )

        self.gaze_mean = nn.Linear(hidden_size//2, 2)    # deterministic actor
        self.gaze_log_std = nn.Parameter(torch.zeros(2))  # log std for gaze action
        self.stop_head = nn.Linear(hidden_size//2, 1)    # stop logit\
        self.stop_head.bias.data.fill_(-2.0)  # negative bias to discourage stopping
        self.value_head = nn.Linear(hidden_size//2, 1)   # state-value

    def forward(self, h):
        x = self.shared(h)
        gaze_mean = torch.tanh(self.gaze_mean(x))
        gaze_std = torch.exp(self.gaze_log_std)
        # gaze_std is a learnable parameter for the model to adjust its own certainty, not a function of h 
        gaze_std = torch.clamp(gaze_std, min=0.01, max=1.0) # Ensure its reasonable range
        stop_logit = self.stop_head(x)
        state_value = self.value_head(x).squeeze(-1)
        return gaze_mean, gaze_std,stop_logit, state_value

    def sample_action(self, h, epsilon=0.1):
        # ε-greedy / noisy continuous action
        gaze_mean, gaze_std, stop_logit, state_value = self.forward(h)
        gaze_dist = torch.distributions.Normal(gaze_mean, gaze_std)
        gaze_action = gaze_dist.sample()  # sample gaze action
        # ensure gaze_action is within [-1, 1] range
        gaze_action = torch.tanh(gaze_action)
        stop_prob = torch.sigmoid(stop_logit)
        stop_action = torch.bernoulli(stop_prob).squeeze(-1)
        return (gaze_action, stop_action), state_value

class Decoder(nn.Module):
    """
    Decoder network that reconstructs the full image purely from spatial memory
    and ignores gaze. Seed removed: decode directly from memory at native resolution.
    """
    def __init__(self, memory_feature_dim=4, memory_size=(32, 32), output_size=(64, 64)):
        super().__init__()
        self.output_size = output_size
        self.memory_size = memory_size
        self.memory_feature_dim = memory_feature_dim

        # Direct decoder from memory (no seed/pooling)
        self.decoder = nn.Sequential(
            nn.Conv2d(memory_feature_dim, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 3, kernel_size=1, stride=1),
            nn.Sigmoid(),
        )

    def forward(self, spatial_memory):
        """
        Reconstruct image from spatial memory only.
        Args:
            spatial_memory: (B, memory_feature_dim, Hm, Wm)
        Returns:
            reconstruction: (B, 3, H, W)
        """
        # Ensure memory matches expected spatial size for the conv path
        if spatial_memory.shape[2:] != self.memory_size:
            spatial_memory = torch.nn.functional.interpolate(
                spatial_memory, size=self.memory_size, mode='bilinear', align_corners=False
            )

        reconstruction = self.decoder(spatial_memory)

        # Ensure output size
        if reconstruction.shape[2:] != self.output_size:
            reconstruction = torch.nn.functional.interpolate(
                reconstruction, size=self.output_size, mode='bilinear', align_corners=False
            )
        return reconstruction

class GazeControlModel(nn.Module):
    """
    The full model with spatial memory core replacing LSTM.
    """
    def __init__(self, encoder_output_size, state_size=512, img_size=(64, 64), fovea_size=(16, 16), memory_size=(32, 32), memory_dim=4):
        super().__init__()
        self.encoder = Encoder(hidden_size=encoder_output_size, input_size=fovea_size)
        self.memory_size = memory_size
        self.encoder_output_size = encoder_output_size
        self.memory_dim = memory_dim

        # Spatial memory core with compact memory representation
        self.spatial_memory_core = SpatialMemoryCore(
            encoder_feature_dim=encoder_output_size,
            memory_dim=memory_dim,
            memory_size=memory_size,
            pos_encoding_dim=64
        )

        # Decoder uses compact memory channels (no gaze input)
        self.decoder = Decoder(
            memory_feature_dim=memory_dim,
            memory_size=memory_size,
            output_size=img_size,
        )

        # Agent now consumes concat(pos_encoding [64], output_gate [576]) => 640-d input
        self.agent = Agent(hidden_size=640)

    def init_memory(self, batch_size, device):
        """Initialize empty spatial cell state."""
        return self.spatial_memory_core.init_cell_state(batch_size, device)

    def forward(self, foveated_image, cell_state, gaze):
        features = self.encoder(foveated_image)
        mem_readout, updated_cell_state = self.spatial_memory_core(features, cell_state, gaze)
        reconstruction = self.decoder(updated_cell_state)  # decoder uses memory only
        pos_encoding = create_spatial_position_encoding(gaze, 64)
        state_input = torch.cat([pos_encoding, mem_readout], dim=1)
        gaze_mean, gaze_std, stop_logit, value = self.agent(state_input)
        return reconstruction, gaze_mean, gaze_std, stop_logit, updated_cell_state

    def sample_action(self, foveated_image, cell_state, gaze):
        features = self.encoder(foveated_image)
        mem_readout, updated_cell_state = self.spatial_memory_core(features, cell_state, gaze)
        reconstruction = self.decoder(updated_cell_state)
        pos_encoding = create_spatial_position_encoding(gaze, 64)
        state_input = torch.cat([pos_encoding, mem_readout], dim=1)
        # Compute policy heads once
        gaze_mean, gaze_std, stop_logit, state_value = self.agent(state_input)
        # Policy-sampled actions
        gaze_dist = torch.distributions.Normal(gaze_mean, gaze_std)
        policy_gaze = torch.tanh(gaze_dist.sample())               # (B,2)
        stop_prob = torch.sigmoid(stop_logit)
        policy_stop = torch.bernoulli(stop_prob).squeeze(-1)       # (B,)
        # Epsilon-greedy: choose random actions with prob ~0.8
        eps = 0.8
        B = gaze_mean.shape[0]
        explore_mask = (torch.rand(B, 1, device=gaze_mean.device) < eps).float()  # (B,1)
        rand_gaze = torch.rand_like(gaze_mean) * 2.0 - 1.0          # uniform in [-1,1]
        rand_stop = torch.bernoulli(torch.full((B,), 0.5, device=gaze_mean.device))  # (B,)
        # Mix actions
        gaze_action = explore_mask * rand_gaze + (1.0 - explore_mask) * policy_gaze  # (B,2)
        stop_action = torch.where(explore_mask.squeeze(1).bool(), rand_stop, policy_stop)  # (B,)
        # Bundle policy heads and explore mask for training
        policy_bundle = (state_value, gaze_mean, gaze_std, stop_logit, explore_mask)
        return reconstruction, (gaze_action, stop_action), policy_bundle, updated_cell_state