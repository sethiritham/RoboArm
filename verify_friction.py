import time
import numpy as np
import pybullet as p
from robot_env import RobotEnv

def verify_friction():
    print("Initializing environment...")
    # Headless mode
    env = RobotEnv(gui=False) 
    env.reset(randomize=False) # Fixed pos

    # Red pos
    # [0.6, -0.1, 0.05]
    red_pos_initial = [0.6, -0.1, 0.05]
    
    print("Moving above Red Cube...")
    env.move_to([0.6, -0.1, 0.3])
    for _ in range(50): env.step()
    
    print("Opening gripper...")
    env.gripper_control(open=True)
    for _ in range(20): env.step()

    print("Descending to grasp...")
    # Descend to 0.03
    # Center=0.025, Grasp=0.03
    env.move_to([0.6, -0.1, 0.03])
    for _ in range(50): env.step()

    print("Closing gripper (applying friction)...")
    env.gripper_control(open=False)
    for _ in range(100): env.step() # Wait to stabilize

    print("Lifting...")
    env.move_to([0.6, -0.1, 0.3])
    for _ in range(100): env.step()

    # Check Red Cube Height
    red_pos, _ = p.getBasePositionAndOrientation(env.redCubeId)
    z_height = red_pos[2]
    
    print(f"Red Cube Final Height: {z_height:.4f}")

    if z_height > 0.2:
        print("SUCCESS: Friction is sufficient. Cube was lifted.")
        result = True
    else:
        print("FAILURE: Cube slipped (friction too low or grip failed).")
        result = False

    env.close()
    return result

if __name__ == "__main__":
    verify_friction()
