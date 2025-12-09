import os
import shutil
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

# Import model
from model import get_model

# Check Colab
try:
    from google.colab import drive
    IN_COLAB = True
except ImportError:
    IN_COLAB = False

# --- 1. SETUP DATA ---
def setup_data(zip_path, extract_path):
    if IN_COLAB:
        print("Mounting Google Drive...")
        drive.mount('/content/drive')
    
    # Unzip (Much faster to read from Colab local disk than Drive)
    if not os.path.exists(extract_path):
        print("Unzipping data...")
        os.makedirs(extract_path, exist_ok=True)
        try:
            shutil.unpack_archive(zip_path, extract_path)
            print("Unzip Complete.")
        except FileNotFoundError:
            print(f"Zip file not found at {zip_path}. Please check the path.")
            return None
    else:
        print("Data already unzipped.")

    # Find data root (if nested)
    data_root = extract_path
    for root, dirs, files in os.walk(extract_path):
        if len(files) > 10: # Found the folder with the .pkl files
            data_root = root
            break
    
    print(f"Data found in: {data_root}")
    return data_root

# --- 2. DATASET ---
class RobotDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.index_map = [] 
        
        if not os.path.exists(data_dir):
             print(f"Data directory {data_dir} does not exist.")
             return

        files = [f for f in os.listdir(data_dir) if f.endswith(".pkl")]
        print(f"Indexing {len(files)} files...")
        
        for filename in files:
            filepath = os.path.join(data_dir, filename)
            try:
                with open(filepath, "rb") as f:
                    data = pickle.load(f)
                    for i in range(len(data)):
                        self.index_map.append((filepath, i))
            except Exception as e:
                print(f"Error reading {filename}: {e}")
                pass
        print(f"Total training frames: {len(self.index_map)}")

    def __len__(self):
        return len(self.index_map)

    def __getitem__(self, idx):
        filepath, local_idx = self.index_map[idx]
        with open(filepath, "rb") as f:
            episode_data = pickle.load(f)
        
        step = episode_data[local_idx]
        
        # Image
        img_pil = Image.fromarray(step['observation'].astype('uint8'), 'RGB')
        if self.transform:
            img_tensor = self.transform(img_pil)
        else:
            # Fallback transform if none provided
            t = transforms.ToTensor()
            img_tensor = t(img_pil)
            
        # Action
        action = np.array(step['action'], dtype=np.float32)
        action_tensor = torch.tensor(action, dtype=torch.float32)
        
        return img_tensor, action_tensor

# --- 3. TRAINING ---
def train_model():
    # Define paths
    if IN_COLAB:
        zip_path = '/content/drive/MyDrive/collected_data.zip'
        extract_path = '/content/temp_data'
        save_model_path = '/content/drive/MyDrive/robot_brain.pth'
    else:
        # Local paths
        zip_path = 'collected_data.zip' 
        extract_path = 'temp_data'
        save_model_path = 'models/robot_brain.pth'

    data_root = setup_data(zip_path, extract_path)
    if not data_root:
        print("Could not find data. Exiting training.")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on: {device}")

    # Transforms
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Load Data
    dataset = RobotDataset(data_root, transform=transform)
    if len(dataset) == 0:
        print("Dataset is empty. Exiting.")
        return
        
    train_loader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=2)

    # Load Model
    model = get_model(device)

    # Loss & Optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    # Loop
    epochs = 15
    print("Starting Training Loop...")

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        
        for images, actions in train_loader:
            images, actions = images.to(device), actions.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, actions)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        
        avg_loss = running_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.4f}")

    # --- 4. SAVE ---
    torch.save(model.state_dict(), save_model_path)
    print(f"Model saved to: {save_model_path}")
    print("Download to run robot!")

if __name__ == "__main__":
    train_model()
