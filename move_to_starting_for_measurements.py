#!/usr/bin/env python3
"""
Utility script to move robot to starting position for measurements.

This script will:
1. Connect to the robot
2. Move to the starting position
3. Hold the position so you can take measurements

Press Ctrl+C to exit and return robot to safe position.
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

# Starting position - same as in main.py
STARTING_POSITION = {
    "shoulder_pan.pos": 0.0,
    "shoulder_lift.pos": -30.0,
    "elbow_flex.pos": -30.0,
    "wrist_flex.pos": 90.0,
    "wrist_roll.pos": 53.0,
    "gripper.pos": 50.0  # Open gripper
}


def smooth_move(robot, target_dict, steps=60, dt=0.04):
    """Smoothly move robot to target position."""
    import numpy as np
    
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
    print("Robot Starting Position Utility")
    print("="*60)
    print("\nThis script will move the robot to the starting position")
    print("so you can take measurements.")
    print("\nStarting position joint values:")
    for joint, value in STARTING_POSITION.items():
        print(f"  {joint:20s}: {value:6.2f}")
    print("\nPress Ctrl+C when done taking measurements to exit.\n")
    
    # Initialize robot
    print("Connecting to robot...")
    config = SO101FollowerConfig(
        port=PORT,
        id=ROBOT_ID,
        calibration_dir=CALIBRATION_DIR,
        cameras={},  # No camera needed for this
        disable_torque_on_disconnect=True,
    )
    
    robot = SO101Follower(config)
    robot.connect(calibrate=False)
    print("Connected!\n")
    
    try:
        # Move to starting position
        print("Moving to starting position...")
        smooth_move(robot, STARTING_POSITION, steps=80, dt=0.04)
        print("\n" + "="*60)
        print("Robot is now at starting position!")
        print("="*60)
        print("\nYou can now take measurements:")
        print("  - Camera height above table")
        print("  - Gripper center position")
        print("  - Arm link lengths")
        print("  - Camera FOV or pixels per inch")
        print("\nRobot will hold this position.")
        print("Press Ctrl+C when done to return to safe position and disconnect.\n")
        
        # Hold position (keep sending commands)
        while True:
            robot.send_action(STARTING_POSITION)
            time.sleep(0.1)  # Send position command every 100ms to hold position
            
    except KeyboardInterrupt:
        print("\n\nInterrupted by user.")
        print("Returning to safe position...")
        
        # Move to a safe position (slightly higher to avoid hitting table)
        safe_position = STARTING_POSITION.copy()
        safe_position["shoulder_lift.pos"] = -40.0  # Lift a bit higher
        safe_position["elbow_flex.pos"] = -40.0     # Retract a bit
        
        smooth_move(robot, safe_position, steps=60, dt=0.04)
        time.sleep(1.0)
        
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        print("Disconnecting robot...")
        robot.disconnect()
        print("Disconnected. Done!")


if __name__ == "__main__":
    main()
