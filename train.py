import os
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from Model import CognitiveModel
import random
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

# Note: All images will be 272x272

# Configuration params maybe moved to a separate config file later
CONFIG = {
    'batch_size': 1,  # For sequential decision-making, batch_size=1 makes most sense
    'lr': 0.0003,  # Learning rate
    'gamma': 0.99,  # Discount factor for RL
    'gru_hidden_size': 512,  # Memory module hidden dimension
    'decoder_hidden_sizes': [256, 128, 64],  # Hidden sizes for upsampling stages
    'decoder_output_size': 3,  # RGB channels
    'ac_hidden_size': 256,  # Actor-critic hidden size
    'num_actions': 8,  # 4 directions + 4 decide actions (one for each quadrant)
    'epochs': 100,
    'entropy_weight': 0.01,  # For encouraging exploration
    'value_loss_weight': 0.5,  # Balancing value and policy losses
    'max_steps_per_episode': 20,  # Maximum glimpses per image to not get stuck in a loop
    'correct_reward': 1.0,  # Reward for correct prediction
    'incorrect_reward': -1.0,  # Penalty for incorrect prediction
    'step_penalty': -0.05,  # Small penalty for each step to encourage efficiency
    'heatmap_decay': 0.95,  # Decay factor for heatmap
    'decoder_loss_weight': 0.2,  # Weight for reconstruction loss
}

class OddOneOutDataset(Dataset):
    def __init__(self, data_dir, split='train', transform=None):
        """
        Args:
            data_dir: Path to data directory
            split: 'train', 'val', or 'test'
            transform: Optional transform to apply to images
        """
        self.data_dir = os.path.join(data_dir, split)
        # Note: Dont resize here like you usually would because we will crop the image later
        self.transform = transform or transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Get all valid image files and their labels
        self.image_files = []
        self.labels = []
        
        for filename in os.listdir(self.data_dir):
            if filename.endswith('.png'):
                # Extract label from filename (something_label.png)
                try:
                    label = int(filename.split('_')[-1].split('.')[0])
                    if 0 <= label <= 3:  # Ensure label is valid
                        self.image_files.append(os.path.join(self.data_dir, filename))
                        self.labels.append(label)
                except (ValueError, IndexError):
                    continue
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        image_path = self.image_files[idx]
        image = Image.open(image_path).convert('RGB')
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
            
        return image, label

class Trainer:
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Setup TensorBoard
        current_time = datetime.now().strftime('%b%d_%H-%M-%S')
        self.log_dir = os.path.join('./runs', f'gazecontrol_{current_time}')
        self.writer = SummaryWriter(log_dir=self.log_dir)
        self.global_step = 0
        
        # Log hyperparameters
        self.writer.add_text('Hyperparameters', str(config))
        
        # Initialize model
        self.model = CognitiveModel(
            gru_hidden_size=config['gru_hidden_size'],
            decoder_hidden_sizes=config['decoder_hidden_sizes'],
            decoder_output_size=config['decoder_output_size'],
            ac_hidden_size=config['ac_hidden_size'],
            num_actions=config['num_actions']
        ).to(self.device)
        
        self.optimizer = optim.Adam(self.model.parameters(), lr=config['lr'])
        
        # Data loaders
        data_dir = "./Data/labeled/"
        self.train_dataset = OddOneOutDataset(data_dir, 'train')
        self.val_dataset = OddOneOutDataset(data_dir, 'val')
        self.test_dataset = OddOneOutDataset(data_dir, 'test')
        
        self.train_loader = DataLoader(
            self.train_dataset, 
            batch_size=1,  # Always 1 for sequential decision making
            shuffle=True
        )
        
        self.val_loader = DataLoader(self.val_dataset, batch_size=1)
        self.test_loader = DataLoader(self.test_dataset, batch_size=1)
    
    def get_fov(self, image, position):
        """Extract limited field of view from the image based on position.
        
        Args:
            image: Full image tensor
            position: (x, y) coordinates (0-1, 0-1) indicating which quarter to view
                      (0,0)=top-left, (1,0)=top-right, (0,1)=bottom-left, (1,1)=bottom-right
        
        Returns:
            Cropped image tensor representing the field of view
        """
        # Divide image into 4 quarters (2x2 grid)
        h, w = image.shape[2], image.shape[3]
        x, y = position
        
        # Calculate crop boundaries for quarters
        crop_h, crop_w = h // 2, w // 2
        top = int(y * crop_h)
        left = int(x * crop_w)
        bottom = top + crop_h
        right = left + crop_w
        
        fov = image[:, :, top:bottom, left:right]

        # Update the Heatmap based on the FOV
                
        # Apply decay to the entire heatmap (memory fades over time)
        self.model.heatmap = self.model.heatmap * self.config['heatmap_decay']
        
        # Create a more subtle Gaussian update centered on the viewed region
        sigma = max(crop_h, crop_w) / 3  # Controls the smoothness

        y_grid, x_grid = torch.meshgrid(
            torch.arange(0, h, device=self.model.heatmap.device),
            torch.arange(0, w, device=self.model.heatmap.device),
            indexing='ij'
        )
        
        # center of the crop
        center_y = top + crop_h // 2
        center_x = left + crop_w // 2
        
        # Create Gaussian weight update
        gaussian = torch.exp(-((x_grid - center_x)**2 + (y_grid - center_y)**2) / (2 * sigma**2))
        
        update_strength = 0.6  # Maximum value to add
        update = gaussian * update_strength
        
        # Update heatmap and clamping to ensure values stay in [0, 1]
        self.model.heatmap = torch.clamp(self.model.heatmap + update, 0.0, 1.0)
        
        # resize the FOV to 224x224 for ResNet
        fov_resized = F.interpolate(fov, size=(224, 224), mode='bilinear', align_corners=False)
        
        return fov_resized
    
    def position_from_action(self, current_position, action):
        """Convert action to new position or decision.
        
        Args:
            current_position: Current (x, y) coordinates (0-1, 0-1)
            action: Integer action:
                   0=left, 1=right, 2=up, 3=down, 
                   4-7=decide quadrant 0-3 is odd
        
        Returns:
            new_position: New (x, y) coordinates
            is_decision: True if the action is a decision, False otherwise
            decision_quadrant: If decision is made, which quadrant was selected (0-3), else None
        """
        x, y = current_position
        
        if action == 0:  # Left
            return (max(0, x-1), y), False, None
        elif action == 1:  # Right
            return (min(1, x+1), y), False, None
        elif action == 2:  # Up
            return (x, max(0, y-1)), False, None
        elif action == 3:  # Down
            return (x, min(1, y+1)), False, None
        else:  # Decide actions (4-7)
            decision_quadrant = action - 4  # Convert to quadrant index (0-3)
            return (x, y), True, decision_quadrant
    
    def train_episode(self, image, label):
        """Train on a single episode (one image)."""
        self.model.reset_memory()
        self.optimizer.zero_grad()
        
        image = image.to(self.device)
        
        # random start location
        position = (random.randint(0, 1), random.randint(0, 1))
        
        # Storage for everything needed for loss calculation
        rewards = []
        log_probs = []
        values = []
        reconstructions = []  
        heatmaps = [] 
        entropy_sum = 0
        
        decision_made = False
        
        # Make sure heatmap is on the correct device
        if self.model.heatmap.device != self.device:
            self.model.heatmap = self.model.heatmap.to(self.device)
        
        for step in range(self.config['max_steps_per_episode']):
            # Get FOV at current position
            fov = self.get_fov(image, position)
            
            # Store a copy of the current heatmap after it's been updated in get_fov
            heatmaps.append(self.model.heatmap.clone())
            
            # Forward pass through model
            action_probs, value, reconstruction = self.model(fov)
            
            # Store reconstruction for decoder loss calculation
            reconstructions.append(reconstruction)
            
            # Calculate entropy for exploration
            dist = torch.distributions.Categorical(action_probs)
            entropy = dist.entropy().mean()
            entropy_sum += entropy
            
            # Sample action
            action = dist.sample()
            log_prob = dist.log_prob(action)
            
            # Move to new position based on action
            position, is_decision, decision_quadrant = self.position_from_action(position, action.item())
            
            # Add step penalty
            reward = self.config['step_penalty']
            
            # If decision is made, check if correct and end episode
            if is_decision:
                decision_made = True
                # Check if the quadrant chosen matches the odd-one-out label
                if decision_quadrant == label:
                    reward += self.config['correct_reward']
                else:
                    reward += self.config['incorrect_reward']
                
                # Store the final decision transition before breaking
                rewards.append(reward)
                log_probs.append(log_prob)
                values.append(value)
                break
            
            # Store transition information
            rewards.append(reward)
            log_probs.append(log_prob)
            values.append(value)
        
        steps_taken = step + 1  # Count steps including the current one
        
        # If no decision was made by max steps, penalize
        if not decision_made:
            rewards[-1] += self.config['incorrect_reward']
        
        # Calculate decoder loss if we have reconstructions
        decoder_loss = torch.tensor(0.0, device=self.device)
        if reconstructions:
            # Image should be 272x272 but just in case
            orig_image = F.interpolate(image, size=(272, 272), mode='bilinear', align_corners=False)
            
            # Calculate weighted reconstruction loss
            total_decoder_loss = 0.0
            for i, recon in enumerate(reconstructions):
                # Get the corresponding heatmap for this step
                step_heatmap = heatmaps[i]
                
                # Calculate pixel-wise MSE loss - shape: [batch_size, 3, 272, 272]
                pixel_loss = F.mse_loss(recon, orig_image, reduction='none')
                
                # Expand heatmap to apply to each channel separately
                # From [272, 272] to [1, 3, 272, 272]
                expanded_heatmap = step_heatmap.unsqueeze(0).unsqueeze(0).expand(-1, 3, -1, -1)
                
                weighted_loss = pixel_loss * expanded_heatmap
                
                # Now average the weighted loss across all dimensions
                step_decoder_loss = weighted_loss.mean()
                total_decoder_loss += step_decoder_loss
            
            # Average over steps
            decoder_loss = total_decoder_loss / len(reconstructions)
            
            # Apply weight from config
            decoder_loss = self.config['decoder_loss_weight'] * decoder_loss
        
        # Calculate returns and advantages for actor-critic
        returns = []
        advantages = []
        R = 0
        
        # Compute returns (discounted sum of rewards)
        for r in reversed(rewards):
            R = r + self.config['gamma'] * R
            returns.insert(0, R)
        
        returns = torch.tensor(returns, device=self.device)
        
        # Check list lengths properly to avoid boolean tensor error
        if len(values) > 0:
            values = torch.cat(values)
            # Make sure returns has same shape as values for the MSE loss
            if returns.shape != values.shape:
                returns = returns.view(values.shape)
            advantages = returns - values.detach()
        
        # Calculate actor-critic losses
        policy_loss = 0
        value_loss = 0
        
        if len(log_probs) > 0:
            log_probs = torch.cat(log_probs)
            policy_loss = -(log_probs * advantages.detach()).mean()
            
        if len(values) > 0:
            value_loss = F.mse_loss(values, returns)
        
        # Average entropy across steps
        steps = len(rewards)
        entropy_loss = -entropy_sum / steps if steps > 0 else 0
        
        # Combine losses
        loss = policy_loss + \
               self.config['value_loss_weight'] * value_loss + \
               self.config['entropy_weight'] * entropy_loss + \
               decoder_loss
        
        
        
        # Log to TensorBoard
        self.writer.add_scalar('Train/policy_loss', policy_loss.item(), self.global_step)
        self.writer.add_scalar('Train/value_loss', value_loss.item(), self.global_step)
        self.writer.add_scalar('Train/entropy_loss', entropy_loss, self.global_step)
        self.writer.add_scalar('Train/decoder_loss', decoder_loss.item(), self.global_step)
        self.writer.add_scalar('Train/total_loss', loss.item(), self.global_step)
        self.writer.add_scalar('Train/episode_reward', sum(rewards), self.global_step)
        self.writer.add_scalar('Train/episode_steps', steps_taken, self.global_step)
        self.writer.add_scalar('Train/correct_prediction', 1 if decision_made and decision_quadrant == label else 0, self.global_step)
        
        # Periodically log visuals (heatmap and reconstruction)
        if self.global_step % 100 == 0 and len(heatmaps) > 0:
            # Log final heatmap
            heatmap_img = heatmaps[-1].cpu().unsqueeze(0)  # Add channel dimension for grayscale
            self.writer.add_image('Train/heatmap', heatmap_img, self.global_step)
            
            # Log reconstruction if available
            if len(reconstructions) > 0:
                # Original image
                orig_img = F.interpolate(image, size=(272, 272), mode='bilinear', align_corners=False)[0].cpu()
                
                # Last reconstruction
                recon_img = reconstructions[-1][0].detach().cpu()
                
                # Combine and log
                comparison = torch.cat([orig_img, recon_img], dim=2)  # Side by side
                self.writer.add_image('Train/original_vs_reconstruction', comparison, self.global_step)
        
        # Increment global step
        self.global_step += 1
        
        loss.backward()
        self.optimizer.step()
        
        return sum(rewards), loss.item(), steps_taken
    
    def train(self):
        """Main training loop using batching and sampling."""
        images_per_epoch = 500
        
        total_epochs = self.config['epochs']
        
        for epoch in range(total_epochs):
            self.model.train()
            total_reward = 0
            total_loss = 0
            total_steps = 0
            correct_decisions = 0
            
            # Create a random subset of the training data for this epoch
            indices = torch.randperm(len(self.train_dataset))[:images_per_epoch]
            
            processed_images = 0
            
            # Process the sampled images
            for idx in indices:
                image, label = self.train_dataset[idx]
                image = image.unsqueeze(0)  # Add batch dimension
                
                reward, loss, steps = self.train_episode(image, label)
                total_reward += reward
                total_loss += loss
                total_steps += steps
                
                # Track correct decisions based on reward
                if reward > self.config['incorrect_reward']:
                    correct_decisions += 1
                
                processed_images += 1
                
                # Print progress every 50 images
                if processed_images % 50 == 0:
                    print(f"Epoch {epoch+1}/{total_epochs} - "
                          f"Progress: {processed_images}/{images_per_epoch} images")
            
            avg_reward = total_reward / images_per_epoch
            avg_loss = total_loss / images_per_epoch
            avg_steps = total_steps / images_per_epoch
            accuracy = correct_decisions / images_per_epoch
            
            # Log epoch-level metrics
            self.writer.add_scalar('Epoch/avg_reward', avg_reward, epoch)
            self.writer.add_scalar('Epoch/avg_loss', avg_loss, epoch)
            self.writer.add_scalar('Epoch/avg_steps', avg_steps, epoch)
            self.writer.add_scalar('Epoch/accuracy', accuracy, epoch)
            
            # Log learning rate
            self.writer.add_scalar('Epoch/learning_rate', self.optimizer.param_groups[0]['lr'], epoch)
            
            print(f"Epoch {epoch+1}/{total_epochs} - " 
                  f"Avg Reward: {avg_reward:.4f}, Avg Loss: {avg_loss:.4f}, "
                  f"Avg Steps: {avg_steps:.2f}, Accuracy: {accuracy:.4f}")
            
            # Run validation every 5 epochs or at the end
            if (epoch + 1) % 5 == 0 or epoch == total_epochs - 1:
                self.validate()
    
    def validate(self):
        """Validate the model on the validation set."""
        self.model.eval()
        total_reward = 0
        correct = 0
        
        with torch.no_grad():
            for image, label in self.val_loader:
                image = image.to(self.device)
                self.model.reset_memory()
                
                # Start at random position
                position = (random.randint(0, 1), random.randint(0, 1))
                
                for step in range(self.config['max_steps_per_episode']):
                    fov = self.get_fov(image, position)
                    
                    # Use deterministic action selection during validation
                    action = self.model.select_action(fov, deterministic=True)
                    
                    # Move to new position
                    position, is_decision, decision_quadrant = self.position_from_action(position, action.item())
                    
                    if is_decision:
                        # Check if decision matches label
                        if decision_quadrant == label.item():
                            correct += 1
                            total_reward += self.config['correct_reward']
                        else:
                            total_reward += self.config['incorrect_reward']
                        break
                    
                    total_reward += self.config['step_penalty']
        
        avg_reward = total_reward / len(self.val_loader)
        accuracy = correct / len(self.val_loader)
        
        print(f"Validation - Avg Reward: {avg_reward:.4f}, Accuracy: {accuracy:.4f}")
    
    def __del__(self):
        """Clean up TensorBoard writer when object is destroyed."""
        if hasattr(self, 'writer'):
            self.writer.close()

if __name__ == "__main__":
    trainer = Trainer(CONFIG)
    print("To view training progress with TensorBoard, run:")
    print("tensorboard --logdir=./runs")
    trainer.train()