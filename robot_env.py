import pybullet as p
import time
import pybullet_data
import math
import numpy as np

class RobotEnv:
    def __init__(self, gui=True):
        # 1. Connect PyBullet
        if gui:
            self.physicsClient = p.connect(p.GUI)
        else:
            self.physicsClient = p.connect(p.DIRECT)
        
        # Search path
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        # Set gravity
        p.setGravity(0, 0, -9.81)

        # Camera configuration
        self.img_width = 224
        self.img_height = 224
        self._configure_camera()

        self.robotId = None
        self.planeId = None
        self.redCubeId = None
        self.blueCubeId = None
        self.num_joints = 0
        self.end_effector_index = 11 # Panda end effector index
        self.gripper_indices = [9, 10] # Panda gripper fingers
        
        print("RobotEnv initialized. Call .reset() to set up the scene.")

    def _configure_camera(self):
        # View Matrix
        self.view_matrix = p.computeViewMatrix(
            cameraEyePosition=[1.0, -0.5, 0.8],    # The camera's position in the world
            cameraTargetPosition=[0.5, 0, 0.1], # The point the camera is looking at
            cameraUpVector=[0, 0, 1]          # The 'up' direction for the camera (Z-axis is up)
        )
        # Projection Matrix
        self.projection_matrix = p.computeProjectionMatrixFOV(
            fov=60.0,
            aspect=float(self.img_width) / self.img_height,
            nearVal=0.1,
            farVal=10.0
        )       

    def reset(self, randomize=False):
        p.resetSimulation()
        p.setGravity(0, 0, -9.81)
        self.planeId = p.loadURDF("plane.urdf")

        # Load Robot
        robotStartPos = [0, 0, 0]
        robotStartOrientation = p.getQuaternionFromEuler([0, 0, 0])
        self.robotId = p.loadURDF("franka_panda/panda.urdf", robotStartPos, robotStartOrientation, useFixedBase=1)
        self.num_joints = p.getNumJoints(self.robotId)

        # --- GRIPPER FRICTION ---
        # Fingers 9, 10
        p.changeDynamics(self.robotId, 9, lateralFriction=2.0, spinningFriction=1.0, rollingFriction=0.0001, frictionAnchor=True)
        p.changeDynamics(self.robotId, 10, lateralFriction=2.0, spinningFriction=1.0, rollingFriction=0.0001, frictionAnchor=True)

        # Reset Pose
        ready_pose = [0, -math.pi/4, 0, -3*math.pi/4, 0, math.pi/2, math.pi/4, 0.04, 0.04]
        index = 0
        for i in range(self.num_joints):
            if p.getJointInfo(self.robotId, i)[3] > -1:
                p.resetJointState(self.robotId, i, ready_pose[index])
                index += 1
                if index >= len(ready_pose): break

        # Load objects
        if randomize:
            red_x = np.random.uniform(0.4, 0.7)
            red_y = np.random.uniform(-0.2, 0.2)
            blue_x = np.random.uniform(0.4, 0.7)
            blue_y = np.random.uniform(-0.2, 0.2)
            while np.linalg.norm(np.array([red_x, red_y]) - np.array([blue_x, blue_y])) < 0.1:
                 blue_x = np.random.uniform(0.4, 0.7)
                 blue_y = np.random.uniform(-0.2, 0.2)
            
            # Spawn high
            red_cube_pos = [red_x, red_y, 0.05]
            blue_cube_pos = [blue_x, blue_y, 0.05]
            red_orn = p.getQuaternionFromEuler([0, 0, np.random.uniform(0, 2 * math.pi)])
            blue_orn = p.getQuaternionFromEuler([0, 0, np.random.uniform(0, 2 * math.pi)])
        else:
            red_cube_pos = [0.6, -0.1, 0.05]
            blue_cube_pos = [0.6, 0.1, 0.05]
            red_orn = p.getQuaternionFromEuler([0, 0, 0])
            blue_orn = p.getQuaternionFromEuler([0, 0, 0])

        self.redCubeId = p.loadURDF("cube.urdf", red_cube_pos, red_orn, globalScaling=0.05)
        self.blueCubeId = p.loadURDF("cube.urdf", blue_cube_pos, blue_orn, globalScaling=0.05)

        # --- CUBE FRICTION ---
        p.changeDynamics(self.redCubeId, -1, lateralFriction=2.0, spinningFriction=1.0, rollingFriction=0.0001, frictionAnchor=True, mass=0.1)
        p.changeDynamics(self.blueCubeId, -1, lateralFriction=2.0, spinningFriction=1.0, rollingFriction=0.0001, frictionAnchor=True, mass=0.1)

        # Colors
        p.changeVisualShape(self.redCubeId, -1, rgbaColor=[1, 0, 0, 1])
        p.changeVisualShape(self.blueCubeId, -1, rgbaColor=[0, 0, 1, 1])

        # Let physics settle
        for _ in range(100): p.stepSimulation()
        
        return self.get_observation()
    def get_cube_positions(self):
        """Returns (x, y, z) of red/blue cubes."""
        red_pos, _ = p.getBasePositionAndOrientation(self.redCubeId)
        blue_pos, _ = p.getBasePositionAndOrientation(self.blueCubeId)
        return list(red_pos), list(blue_pos)
    
    def move_to(self, target_pos, target_orn=None):
        """
        Moves robot EE to pos/orn.
        """
        if target_orn is None:
            # Default orn: down
            target_orn = p.getQuaternionFromEuler([math.pi, 0, 0])
            
        joint_poses = p.calculateInverseKinematics(
            self.robotId,
            self.end_effector_index,
            target_pos,
            target_orn
        )
        
        # Apply joint poses
        index = 0
        for i in range(self.num_joints):
            joint_info = p.getJointInfo(self.robotId, i)
            q_index = joint_info[3]
            if q_index > -1:
                p.setJointMotorControl2(
                    bodyUniqueId=self.robotId,
                    jointIndex=i,
                    controlMode=p.POSITION_CONTROL,
                    targetPosition=joint_poses[index]
                )
                index += 1
                
    def gripper_control(self, open=True):
        """
        Control gripper.
        """
        target_pos = 0.04 if open else 0.0 # 0.04=Open, 0.0=Closed
        for i in self.gripper_indices:
             p.setJointMotorControl2(
                bodyUniqueId=self.robotId,
                jointIndex=i,
                controlMode=p.POSITION_CONTROL,
                targetPosition=target_pos,
                force=100
            )

    def step(self, action=None):
        # Action handling if needed
        
        # Advance simulation
        p.stepSimulation()
        
        # Get new observation
        obs = self.get_observation()
        
        reward = 0
        done = False
        info = {}
        
        return obs, reward, done, info

    def get_observation(self):
        # Capture cam image
        width, height, rgb_pixels, depth_pixels, segmentation_mask = p.getCameraImage(
            width=self.img_width,
            height=self.img_height,
            viewMatrix=self.view_matrix,
            projectionMatrix=self.projection_matrix,
            renderer=p.ER_BULLET_HARDWARE_OPENGL 
        )
        
        image_array = np.array(rgb_pixels).reshape((height, width, 4))
        rgb_image = image_array[:, :, :3]
        
        return rgb_image

    def close(self):
        p.disconnect()
        print("Disconnected from PyBullet.")

# --- Demo Usage of the RobotEnv Class ---
if __name__ == "__main__":
    env = RobotEnv(gui=True)
    obs = env.reset()
    
    print("Moving to Blue Cube...")
    # High-level script for testing
    # 1. Above blue
    env.move_to([0.6, 0.1, 0.3])
    for _ in range(100): env.step()
    
    # 2. Open
    env.gripper_control(open=True)
    for _ in range(50): env.step()
    
    # 3. Descend
    env.move_to([0.6, 0.1, 0.025])
    for _ in range(100): env.step()
    
    # 4. Close
    env.gripper_control(open=False)
    for _ in range(100): env.step()
    
    # 5. Ascend
    env.move_to([0.6, 0.1, 0.3])
    for _ in range(100): env.step()
    
    print("Moving to Red Cube...")
    # 6. Above red
    env.move_to([0.6, -0.1, 0.3])
    for _ in range(100): env.step()
    
    # 7. Descend
    env.move_to([0.6, -0.1, 0.075])
    for _ in range(100): env.step()
    
    # 8. Open
    env.gripper_control(open=True)
    for _ in range(50): env.step()
    
    print("Demo sequence complete. Press Ctrl+C to exit.")
    try:
        while True:
            env.step()
            time.sleep(1./240.)
    except KeyboardInterrupt:
        env.close()