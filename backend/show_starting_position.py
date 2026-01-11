#!/usr/bin/env python3
"""
Utility script to move robot to starting position and display camera feed.

This script will:
1. Connect to the robot
2. Move to the starting position
3. Display the camera feed
4. Hold the position until Ctrl+C is pressed
"""

from datetime import datetime
import time
import os
import cv2
import numpy as np
from pathlib import Path
from dotenv import load_dotenv
from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig
from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig

load_dotenv()

PORT = os.getenv("ROBOT_PORT", "/dev/tty.usbmodem5A7C1217691")
CALIBRATION_DIR = Path(__file__).parent
ROBOT_ID = "hackathon_robot"

# Starting position - same as in main.py
STARTING_POSITION = {
    "shoulder_pan.pos": 6.23,
    "shoulder_lift.pos": -35.33,
    "elbow_flex.pos": 17.29,
    "wrist_flex.pos": 86.77,
    "wrist_roll.pos": 52.43,
    "gripper.pos": 49.81,
}


def smooth_move(robot, target_dict, steps=60, dt=0.04):
    """Smoothly move robot to target position."""
    # Get current positions
    obs = robot.get_observation()
    joint_names = list(robot.bus.motors.keys())
    start_dict = {f"{name}.pos": obs[f"{name}.pos"] for name in joint_names}
    
    # Interpolate from start to target
    for a in np.linspace(0, 1, steps):
        action = {
            key: (1 - a) * start_dict[key] + a * target_dict[key]
            for key in target_dict.keys()
        }
        robot.send_action(action)
        time.sleep(dt)


def main():
    print("="*60)
    print("Robot Starting Position with Camera View")
    print("="*60)
    print("\nThis script will move the robot to the starting position")
    print("and display the camera feed.")
    print("\nStarting position joint values:")
    for joint, value in STARTING_POSITION.items():
        print(f"  {joint:20s}: {value:6.2f}")
    print("\nPress Ctrl+C to exit.\n")
    
    # Initialize robot with camera
    print("Connecting to robot...")
    camera_config = {
        "wrist": OpenCVCameraConfig(index_or_path=0, width=640, height=480, fps=30)
    }
    
    config = SO101FollowerConfig(
        port=PORT,
        id=ROBOT_ID,
        calibration_dir=CALIBRATION_DIR,
        cameras=camera_config,
        disable_torque_on_disconnect=True,
    )
    
    robot = SO101Follower(config)
    
    try:
        robot.connect(calibrate=False)
        print("Connected!\n")
    except RuntimeError as e:
        if "motor check failed" in str(e) or "Missing motor" in str(e):
            print("\n" + "="*60)
            print("ERROR: Cannot connect to robot!")
            print("="*60)
            print("\nPossible issues:")
            print("  1. Robot is not powered on")
            print("  2. USB cable is not connected")
            print("  3. Wrong port (current: {})".format(PORT))
            print("  4. Another program is using the robot")
            print("\nPlease check:")
            print("  - Is the robot powered on?")
            print("  - Is the USB cable connected?")
            print("  - Is the port correct? (check .env file)")
            print("  - Try closing other programs that might be using the robot")
            print("="*60 + "\n")
        raise
    
    try:
        # Move to starting position
        print("Moving to starting position...")
        smooth_move(robot, STARTING_POSITION, steps=80, dt=0.04)
        print("\n" + "="*60)
        print("Robot is now at starting position!")
        print("Camera feed will be displayed.")
        print("Press Ctrl+C to exit.")
        print("="*60 + "\n")
        
        # Display camera feed
        camera_key = "wrist"
        window_name = "Robot Camera Feed - Starting Position"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

        # Take screenshot
        print("Capturing screenshot...")
        images_dir = Path(__file__).parent / "images"
        images_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        screenshot_path = images_dir / f"starting_position_{timestamp}.png"
        
        # Get one frame and save it
        obs = robot.get_observation()
        if "wrist" in obs and isinstance(obs["wrist"], np.ndarray):
            frame = obs["wrist"]
            if len(frame.shape) == 3 and frame.shape[2] == 3:
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                cv2.imwrite(str(screenshot_path), frame_bgr)
                print(f"Screenshot saved to: {screenshot_path}\n")
        
        while True:
            # Get camera frame
            obs = robot.get_observation()
            if camera_key in obs and isinstance(obs[camera_key], np.ndarray):
                frame = obs[camera_key]
                if len(frame.shape) == 3 and frame.shape[2] == 3:
                    # Convert RGB to BGR for OpenCV
                    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                    cv2.imshow(window_name, frame_bgr)
            
            # Hold position (send command every 100ms)
            robot.send_action(STARTING_POSITION)
            
            # Check for exit (ESC or window close)
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC key
                print("\nESC pressed. Exiting...")
                break
            
            time.sleep(0.1)  # Small delay for camera and position hold
            
    except KeyboardInterrupt:
        print("\n\nInterrupted by user.")
        
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        cv2.destroyAllWindows()
        print("Disconnecting robot...")
        robot.disconnect()
        print("Disconnected. Done!")


if __name__ == "__main__":
    main()
