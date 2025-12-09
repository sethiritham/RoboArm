import time
import numpy as np
import pybullet as p
from robot_env import RobotEnv
from dataset_recorder import DatasetRecorder

def main():
    # Headless for speed; toggle for view
    env = RobotEnv(gui=False)
    recorder = DatasetRecorder(save_dir="collected_data")
    
    # 1000 episodes for data
    num_episodes = 1000
    
    for episode in range(num_episodes):
        print(f"Starting Episode {episode + 1}/{num_episodes}")
        
        # Randomize start
        obs = env.reset(randomize=True)
        recorder.reset_buffer()
        
        # Get positions
        red_pos, blue_pos = env.get_cube_positions()
        
        # --- HEIGHTS ---
        # Cube=0.05, Floor=0
        
        safe_z = 0.3
        
        # Pick: center of blue
        # 0.05 / 2 = 0.025
        pick_z = 0.025 
        
        # Place: stack on red (0.05 + 0.025)
        place_z = 0.075 
        
        waypoints = [
            # 1. Above Blue
            {"pos": [blue_pos[0], blue_pos[1], safe_z], "grip": 1},
            # 2. Descend Blue
            {"pos": [blue_pos[0], blue_pos[1], pick_z], "grip": 1},
            # 3. Pick
            {"pos": [blue_pos[0], blue_pos[1], pick_z], "grip": 0},
            # 4. Ascend
            {"pos": [blue_pos[0], blue_pos[1], safe_z], "grip": 0},
            # 5. Above Red
            {"pos": [red_pos[0], red_pos[1], safe_z], "grip": 0},
            # 6. Stack
            {"pos": [red_pos[0], red_pos[1], place_z], "grip": 0},
            # 7. Release
            {"pos": [red_pos[0], red_pos[1], place_z], "grip": 1},
            # 8. Ascend (Done)
            {"pos": [red_pos[0], red_pos[1], safe_z], "grip": 1},
        ]
        
        for step_idx, wp in enumerate(waypoints):
            target_pos = wp["pos"]
            target_grip = wp["grip"] # 1: Open, 0: Close
            
            # Move
            env.move_to(target_pos)
            env.gripper_control(open=(target_grip == 1))
            
            # Settle & record checking
            steps_per_waypoint = 50
            for i in range(steps_per_waypoint):
                env.step() # Advance physics
                
                # Save every 10th frame (trajectory data)
                if i % 10 == 0:
                    current_obs = env.get_observation()
                    # Action: target pos + grip
                    action = target_pos + [target_grip]
                    
                    reward = 0
                    done = False
                    recorder.add_step(current_obs, action, reward, done)
        
        # Save
        recorder.save_episode()
        # print(f"Episode {episode + 1} complete.") 

    env.close()

if __name__ == "__main__":
    main()