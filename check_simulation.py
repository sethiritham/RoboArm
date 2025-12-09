import pybullet as p
import time
import pybullet_data
import os

print("PyBullet simulation environment check...")

# 1. Init PyBullet (GUI or DIRECT)
physicsClient = p.connect(p.GUI) 

# 2. Add data path
p.setAdditionalSearchPath(pybullet_data.getDataPath())

# 3. Set gravity
p.setGravity(0, 0, -9.81)

# 4. Load plane
planeId = p.loadURDF("plane.urdf")

# 5. Load cube (high)
cubeStartPos = [0, 0, 1]
cubeStartOrientation = p.getQuaternionFromEuler([0, 0, 0])
boxId = p.loadURDF("cube.urdf", cubeStartPos, cubeStartOrientation)

print("Simulation started. A plane and a cube should be visible if in GUI mode.")
print("The cube will fall due to gravity.")

# 6. Simulate 16s
for i in range(1000): # 1000 steps * 1/60s
    p.stepSimulation()
    time.sleep(1./240.) # Smaller sleep for smoother GUI, adjust as needed

print("Simulation finished.")

# 7. Disconnect
p.disconnect()
print("PyBullet environment check complete.")

# User info
print("\nIf you saw a window with a gray plane and a red cube falling onto it,")
print("then PyBullet is working correctly!")
print("If not, check 'pybullet' install/drivers.")
