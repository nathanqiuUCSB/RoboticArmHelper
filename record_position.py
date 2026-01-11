#!/usr/bin/env python3
"""
Utility script to capture the current robot position for STAGE 2 starting position.

Instructions:
1. Manually position the robot to your desired STAGE 2 starting position
2. Run this script: python capture_stage2_starting_position.py
3. The script will read and display the current joint positions
4. Copy the output into main.py as STAGE2_STARTING_POSITION_TEMPLATE
"""

import time
import os
from pathlib import Path
from dotenv import load_dotenv
from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig

load_dotenv()

PORT = os.getenv("ROBOT_PORT", "/dev/tty.usbmodem5A7C1217691")
CALIBRATION_DIR = Path(__file__).parent
ROBOT_ID = "hackathon_robot"


def main():
    print("="*60)
    print("STAGE 2 Starting Position Capture Utility")
    print("="*60)
    print("\nInstructions:")
    print("1. Manually position the robot to your desired STAGE 2 starting position")
    print("2. Make sure the robot is positioned correctly")
    print("3. The script will read the current joint positions")
    print("\nPress Enter when ready to capture the position...")
    input()
    
    # Connect to robot
    print("\nConnecting to robot...")
    config = SO101FollowerConfig(
        port=PORT,
        id=ROBOT_ID,
        calibration_dir=CALIBRATION_DIR,
        cameras={},
        disable_torque_on_disconnect=True,
    )
    
    robot = SO101Follower(config)
    robot.connect(calibrate=False)
    
    try:
        # Read current positions
        print("Reading current joint positions...")
        time.sleep(0.5)  # Give it a moment to stabilize
        obs = robot.get_observation()
        
        joint_names = list(robot.bus.motors.keys())
        positions = {f"{name}.pos": obs[f"{name}.pos"] for name in joint_names}
        
        print("\n" + "="*60)
        print("CAPTURED JOINT POSITIONS:")
        print("="*60)
        print("\n# STAGE 2 Starting Position Template")
        print("STAGE2_STARTING_POSITION_TEMPLATE = {")
        for key in ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]:
            pos_key = f"{key}.pos"
            if pos_key in positions:
                value = positions[pos_key]
                print(f'    "{pos_key}": {value:.2f},')
        print("}")
        
        print("\n" + "="*60)
        print("Copy the dictionary above into main.py as STAGE2_STARTING_POSITION_TEMPLATE")
        print("="*60)
        
    except KeyboardInterrupt:
        print("\nInterrupted by user.")
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
    finally:
        robot.disconnect()
        print("\nRobot disconnected.")


if __name__ == "__main__":
    main()
