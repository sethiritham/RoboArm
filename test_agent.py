import time
import torch
import numpy as np
from torchvision import models, transforms
from PIL import Image
from robot_env import RobotEnv

# --- CONFIGURATION ---
MODEL_PATH = "models/robot_brain.pth" # Model path
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- 1. SETUP MODEL ---
def load_brain():
    print(f"Loading brain from {MODEL_PATH}...")
    
    # Re-create architecture
    model = models.resnet18(weights=None) 
    num_ftrs = model.fc.in_features
    model.fc = torch.nn.Linear(num_ftrs, 4) # X, Y, Z, Grip
    
    # Load weights (CPU safe)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval() # Eval mode
    model.to(device)
    
    print("Brain loaded successfully.")
    return model

# --- 2. IMAGE PRE-PROCESSING ---
# Match training transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def main():
    # Init Env (GUI)
    env = RobotEnv(gui=True)
    model = load_brain()
    
    print("\n--- STARTING AI CONTROL ---")
    print("Robot acts on vision.")
    
    # Run 5 demo episodes
    for episode in range(5):
        print(f"Episode {episode+1} starting...")
        obs = env.reset(randomize=True)
        
        # Run the loop
        for step in range(500): # Max steps per episode
            # 1. Process Image
            # Numpy -> PIL -> Tensor
            img_pil = Image.fromarray(obs.astype('uint8'), 'RGB')
            img_tensor = transform(img_pil).unsqueeze(0) # Batch
            img_tensor = img_tensor.to(device)
            
            # 2. Consult the Brain
            with torch.no_grad(): # Inference
                output = model(img_tensor).cpu().numpy()[0]
            
            # 3. Decode Action
            target_pos = output[:3]
            gripper_prob = output[3]
            
            # Threshold: >0.5 Open, <0.5 Close
            open_gripper = gripper_prob > 0.5
            
            # 4. Execute
            env.move_to(target_pos)
            env.gripper_control(open=open_gripper)
            
            # 5. Step Simulation
            obs, _, _, _ = env.step()
            
            # Sleep for view
            time.sleep(0.01)
            
    print("Demo Complete.")
    env.close()

if __name__ == "__main__":
    main()
