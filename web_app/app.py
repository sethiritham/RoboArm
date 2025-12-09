import sys
import os
import cv2
import time
import threading
from flask import Flask, render_template, Response, jsonify

# Add parent directory to path to import robot_env
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from robot_env import RobotEnv

app = Flask(__name__)

# Global Environment
# We use gui=False because we don't want the PyBullet window to pop up on the server side
# We will render images ourselves and stream them.
env = RobotEnv(gui=False)
env.reset()

frame_buffer = None
lock = threading.Lock()

def simulation_loop():
    """
    Background thread to step the simulation and update the frame buffer.
    """
    global frame_buffer
    while True:
        with lock:
            # Step simulation
            obs, _, _, _ = env.step()
            
            # Convert RGB to BGR for OpenCV/JPEG
            bgr_image = cv2.cvtColor(obs, cv2.COLOR_RGB2BGR)
            
            # Encode as JPEG
            ret, buffer = cv2.imencode('.jpg', bgr_image)
            frame_buffer = buffer.tobytes()
        
        time.sleep(1./30.) # 30 FPS

# Start simulation thread
t = threading.Thread(target=simulation_loop, daemon=True)
t.start()

@app.route('/')
def index():
    return render_template('index.html')

def generate():
    global frame_buffer
    while True:
        with lock:
            if frame_buffer is None:
                continue
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_buffer + b'\r\n')
        time.sleep(0.01)

@app.route('/video_feed')
def video_feed():
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/control/<action>', methods=['POST'])
def control(action):
    # Map actions to robot commands
    # This interacts with the global env
    print(f"Received action: {action}")
    
    with lock:
        if action == "reset":
            env.reset()
        elif action == "pick_blue":
            # Simple scripted action for demo
            # In a real app, this might trigger a state machine or policy
            env.move_to([0.6, 0.1, 0.3])
            env.gripper_control(open=True)
            for _ in range(50): env.step()
            env.move_to([0.6, 0.1, 0.05])
            for _ in range(50): env.step()
            env.gripper_control(open=False)
            for _ in range(50): env.step()
            env.move_to([0.6, 0.1, 0.3])
            for _ in range(50): env.step()
            
        elif action == "place_on_red":
             env.move_to([0.6, -0.1, 0.3])
             for _ in range(50): env.step()
             env.move_to([0.6, -0.1, 0.09])
             for _ in range(50): env.step()
             env.gripper_control(open=True)
             for _ in range(50): env.step()
             env.move_to([0.6, -0.1, 0.3])
             for _ in range(50): env.step()
             
    return jsonify({"status": "success", "action": action})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
