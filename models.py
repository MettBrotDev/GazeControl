import torch
import torch.nn as nn

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
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1), # H/4xW/4 -> H/4xW/4 (no stride)
            nn.ReLU(),
        )
        
        # Calculate FC layer size based on input_size
        # After 2 conv layers with stride 2, feature map size is input_size / 4
        feature_h, feature_w = input_size[0] // 4, input_size[1] // 4
        fc_input_size = 64 * feature_h * feature_w
        self.fc = nn.Linear(fc_input_size, hidden_size)

    def forward(self, x):
        x = self.cnn(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

class RecurrentCore(nn.Module):
    """
    Recurrent core (LSTM) that processes features and outputs a new hidden state.
    """
    def __init__(self, input_size, hidden_size=512):
        super().__init__()
        self.lstm_cell = nn.LSTMCell(input_size, hidden_size)

    def forward(self, features, h_prev, c_prev, gaze):
        # Concatenate image features with gaze coordinates
        lstm_input = torch.cat((features, gaze), dim=1)
        h_new, c_new = self.lstm_cell(lstm_input, (h_prev, c_prev))
        return h_new, c_new

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
    Decoder network that reconstructs the full image from the hidden state.
    Input: (B, 512)
    Output: (B, 3, IMG_SIZE[0], IMG_SIZE[1]) - matches Config.IMG_SIZE
    """
    def __init__(self, hidden_size=512, output_size=(256, 256)):
        super().__init__()
        self.output_size = output_size
        
        # Calculate the starting feature map size based on output size
        # We'll use 4 upsampling layers (each 2x), so we need output_size / 16
        start_h = output_size[0] // 16
        start_w = output_size[1] // 16
        
        self.fc = nn.Linear(hidden_size, 256 * start_h * start_w)
        self.start_h = start_h
        self.start_w = start_w
        
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1), # 2x upsampling
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),  # 2x upsampling
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),   # 2x upsampling
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, kernel_size=4, stride=2, padding=1),    # 2x upsampling
            nn.Sigmoid() # Output pixel values between 0 and 1
        )

    def forward(self, h):
        x = self.fc(h)
        x = x.view(x.size(0), 256, self.start_h, self.start_w)
        x = self.deconv(x)
        
        # Ensure output matches exactly the target size
        if x.shape[2:] != self.output_size:
            x = torch.nn.functional.interpolate(x, size=self.output_size, mode='bilinear', align_corners=False)
        
        return x

class GazeControlModel(nn.Module):
    """
    The full model combining Encoder, RecurrentCore, Decoder, and Agent.
    """
    def __init__(self, encoder_output_size, hidden_size=512, img_size=(256, 256), fovea_size=(16, 16)):
        super().__init__()
        self.encoder = Encoder(hidden_size=encoder_output_size, input_size=fovea_size)
        # The input to the recurrent core is the feature vector from the encoder
        # plus the 2 gaze coordinates.
        self.recurrent_core = RecurrentCore(input_size=encoder_output_size + 2, hidden_size=hidden_size)
        self.decoder = Decoder(hidden_size=hidden_size, output_size=img_size)
        self.agent = Agent(hidden_size=hidden_size)

    def forward(self, foveated_image, h_prev, c_prev, gaze):
        features = self.encoder(foveated_image)
        h_new, c_new = self.recurrent_core(features, h_prev, c_prev, gaze)
        reconstruction = self.decoder(h_new)
        gaze_mean, gaze_std, stop_logit, value = self.agent(h_new)
        return reconstruction, gaze_mean, gaze_std, stop_logit, (h_new, c_new)
    
    def sample_action(self, foveated_image, h_prev, c_prev, gaze):
        # Returns: reconstruction, action, state_value, (h_new, c_new)
        features = self.encoder(foveated_image)
        h_new, c_new = self.recurrent_core(features, h_prev, c_prev, gaze)
        reconstruction = self.decoder(h_new)
        action, state_value = self.agent.sample_action(h_new)
        return reconstruction, action, state_value, (h_new, c_new)