import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
import numpy as np


'''Feature extractor for the model

** Using pretrained ResNet50 model right now'''

class FeatureExtractor(nn.Module):
    def __init__(self):
        super(FeatureExtractor, self).__init__()
        # Update to use modern weights parameter
        from torchvision.models import ResNet50_Weights
        self.resnet = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        # Remove the last fully connected layer to only keep the feature extractor part
        self.resnet = nn.Sequential(*list(self.resnet.children())[:-1])
        # Set the model to evaluation mode
        self.resnet.eval()


    def forward(self, x):
        # No grad since we just use the pretrained model for now
        with torch.no_grad():
            features = self.resnet(x)
        return features.squeeze()
    
'''Memory module for the model

** Using a GRU for now. Might look into other options later'''

class MemoryModule(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(MemoryModule, self).__init__()
        self.gru = nn.GRU(input_size, hidden_size, batch_first=True)
        self.hidden_size = hidden_size

    def forward(self, x, h):
        # Return both output and hidden state for simpler usage
        out, h_new = self.gru(x, h)
        return out, h_new
    
'''Decoder module for the model
Input: GRU output
Output: tensor with shape [batch_size, 3, 272, 272] containing normalized RGB values (0-1)
** Using a CNN decoder for now. Might look into other options later'''
class DecoderModule(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size=3):
        super(DecoderModule, self).__init__()
        
        # Starting from 1x1, we need to reach 272x272
        # 1 -> 17 -> 34 -> 68 -> 136 -> 272
        
        self.initial_projection = nn.Sequential(
            nn.Linear(input_size, hidden_sizes[0] * 17 * 17),
            nn.ReLU()
        )
        
        # Transposed convolutions for upsampling
        self.upsampling = nn.Sequential(
            # 17x17 -> 34x34
            nn.ConvTranspose2d(hidden_sizes[0], hidden_sizes[0], kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(hidden_sizes[0]),
            
            # 34x34 -> 68x68
            nn.ConvTranspose2d(hidden_sizes[0], hidden_sizes[1], kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(hidden_sizes[1]),
            
            # 68x68 -> 136x136
            nn.ConvTranspose2d(hidden_sizes[1], hidden_sizes[2], kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(hidden_sizes[2]),
            
            # 136x136 -> 272x272
            nn.ConvTranspose2d(hidden_sizes[2], output_size, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid()  # Use sigmoid to constrain values between 0 and 1
        )
    
    def forward(self, x):
        # x has shape [batch_size, input_size]
        batch_size = x.size(0)
        
        # Project and reshape to initial spatial dimensions
        x = self.initial_projection(x)
        x = x.view(batch_size, -1, 17, 17)
        
        # Apply upsampling layers to create a tensor with normalized RGB values
        # Output shape: [batch_size, 3, 272, 272] with values in range [0,1]
        x = self.upsampling(x)
        
        return x
        
'''Reinforcement module for the model
Input: GRU output
Output: action (either move camera or lock in a final guess)
** Using a simple Actor critic.'''
class ActorCritic(nn.Module):
    def __init__(self, input_size, hidden_size, num_actions):
        super(ActorCritic, self).__init__()
        
        # Shared feature layer for both of them to analyze the current state
        self.shared_layers = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU()
        )
        
        # Actor network - outputs action probabilities
        self.actor = nn.Sequential(
            nn.Linear(hidden_size, hidden_size//2),
            nn.ReLU(),
            nn.Linear(hidden_size//2, num_actions),
            nn.Softmax(dim=-1)
        )
        
        # Critic network - estimates value function
        self.critic = nn.Sequential(
            nn.Linear(hidden_size, hidden_size//2),
            nn.ReLU(),
            nn.Linear(hidden_size//2, 1)
        )
    
    def forward(self, x):
        shared_features = self.shared_layers(x)
        
        # Get action probabilities from actor
        action_probs = self.actor(shared_features)
        
        # Get value estimate from critic
        value = self.critic(shared_features)
        
        return action_probs, value
    
    def select_action(self, state, deterministic=False):
        with torch.no_grad():
            action_probs, _ = self.forward(state)
            
            if deterministic:
                # Choose the most probable action
                action = torch.argmax(action_probs, dim=-1)
            else:
                # Sample from the probability distribution
                dist = torch.distributions.Categorical(action_probs)
                action = dist.sample()
                
        return action

'''Combined model that integrates all components:
- Feature extractor processes images
- Memory module (GRU) maintains state
- Output splits to both decoder (reconstruction) and actor-critic (actions)
'''
class CognitiveModel(nn.Module):
    def __init__(self, gru_hidden_size, decoder_hidden_sizes, decoder_output_size, ac_hidden_size, num_actions):
        super(CognitiveModel, self).__init__()
        
        # Feature extractor (ResNet50)
        self.feature_extractor = FeatureExtractor()
        
        # Automatically detect feature size from the feature extractor
        with torch.no_grad():
            dummy_input = torch.zeros(1, 3, 224, 224)  # Standard input size
            features = self.feature_extractor(dummy_input)
            self.feature_size = features.shape[-1]  # Get the feature dimension
        
        # Memory module (GRU)
        self.memory = MemoryModule(self.feature_size, gru_hidden_size)
        
        # Decoder for image reconstruction (now properly configured for 272x272)
        self.decoder = DecoderModule(gru_hidden_size, decoder_hidden_sizes, decoder_output_size)
        
        # Actor-Critic for action selection
        self.actor_critic = ActorCritic(gru_hidden_size, ac_hidden_size, num_actions)
        
        # Initial hidden state
        self.h0 = None

        # Initial Heatmap (272x272)
        # The heatmap should always be between 0 and 1 and gives information on what the model should already know.
        # It should always be between 0 and 1 for each pixel value. We will later use this to calculate the decoder loss.
        self.heatmap = torch.zeros((272, 272), dtype=torch.float32)
        
    def reset_memory(self):
        """Reset the memory/hidden state of the GRU"""
        device = next(self.parameters()).device
        self.h0 = torch.zeros(1, 1, self.memory.hidden_size, device=device)
        
    def forward(self, x, return_reconstruction=True):
        """
        Forward pass through the model
        Args:
            x: Input image
            return_reconstruction: Whether to return the reconstructed image
        
        Returns:
            action_probs: Action probabilities from actor
            value: Value estimate from critic
            reconstruction: Reconstructed image (if return_reconstruction=True)
        """
        # Always batch size 1 for simplicity
        
        # Initialize hidden state if not already done
        if self.h0 is None:
            self.reset_memory()
            
        # Extract features
        features = self.feature_extractor(x)
        
        # Reshape for GRU if needed
        if len(features.shape) == 1:
            features = features.unsqueeze(0).unsqueeze(0)  # Add batch and seq dims
        elif len(features.shape) == 2:
            features = features.unsqueeze(1)  # Add sequence dimension
            
        # Pass through memory module with simplified handling
        memory_output, self.h0 = self.memory(features, self.h0)
        
        # Get final memory output
        final_mem_output = memory_output[:, -1]  # Shape: (1, gru_hidden_size)
        
        # Split to decoder and actor-critic
        action_probs, value = self.actor_critic(final_mem_output)  # Actor-critic takes 2D input
        
        if return_reconstruction:
            # Generate tensor of normalized RGB values (shape: [batch_size, 3, 272, 272])
            reconstruction = self.decoder(final_mem_output)
            return action_probs, value, reconstruction
        else:
            return action_probs, value
    
    def select_action(self, state, deterministic=False):
        """Wrapper for actor-critic's select_action"""
        # Extract features and update memory
        with torch.no_grad():
            features = self.feature_extractor(state)
            if len(features.shape) == 1:
                features = features.unsqueeze(0).unsqueeze(0)  # Add batch and seq dims
            elif len(features.shape) == 2:
                features = features.unsqueeze(1)  # Add sequence dimension
                
            memory_output, self.h0 = self.memory(features, self.h0)
            
            final_mem_output = memory_output[:, -1]
            
            return self.actor_critic.select_action(final_mem_output, deterministic)
