import os
import numpy as np
import pickle
from datetime import datetime

class DatasetRecorder:
    def __init__(self, save_dir="dataset"):
        self.save_dir = save_dir
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        self.current_episode = []
    
    def add_step(self, observation, action, reward, done):
        """
        Add step to buffer.
        """
        step_data = {
            "observation": observation,
            "action": action,
            "reward": reward,
            "done": done
        }
        self.current_episode.append(step_data)

    def save_episode(self):
        """
        Save buffer to file.
        """
        if not self.current_episode:
            print("No data to save.")
            return

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"episode_{timestamp}.pkl"
        filepath = os.path.join(self.save_dir, filename)
        
        with open(filepath, "wb") as f:
            pickle.dump(self.current_episode, f)
        
        print(f"Saved episode with {len(self.current_episode)} steps to {filepath}")
        self.current_episode = [] # Reset buffer

    def reset_buffer(self):
        self.current_episode = []
