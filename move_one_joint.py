import time
import os
from pathlib import Path
from dotenv import load_dotenv
import numpy as np
from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig

load_dotenv()

PORT = os.getenv("ROBOT_PORT", "/dev/tty.usbmodem5A7C1217691")  # Default port if not in .env

# Calibration file location
# The calibration file should be named {robot_id}.json in the calibration_dir
CALIBRATION_DIR = Path(__file__).parent  # Current directory where hackathon_robot.json is
ROBOT_ID = "hackathon_robot"  # This will look for hackathon_robot.json


def smooth_send(robot, start_dict, target_dict, steps=40, dt=0.02):
    """Smoothly interpolate between two action dictionaries."""
    joint_keys = list(start_dict.keys())
    for a in np.linspace(0, 1, steps):
        # Interpolate each joint position
        action = {
            key: (1 - a) * start_dict[key] + a * target_dict[key]
            for key in joint_keys
        }
        robot.send_action(action)
        time.sleep(dt)


# Create robot configuration
config = SO101FollowerConfig(
    port=PORT,
    id=ROBOT_ID,  # This tells lerobot to load hackathon_robot.json from calibration_dir
    calibration_dir=CALIBRATION_DIR,  # Directory containing hackathon_robot.json
    cameras={},  # No cameras needed for this test
    disable_torque_on_disconnect=True,
)

# Create and connect robot
robot = SO101Follower(config)
robot.connect(calibrate=False)

# Get current observation - returns dict with keys like "shoulder_pan.pos", etc.
obs = robot.get_observation()

# Extract joint positions into a dict
# The observation has keys like "shoulder_pan.pos", "shoulder_lift.pos", etc.
joint_names = list(robot.bus.motors.keys())  # ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
q0_dict = {f"{name}.pos": obs[f"{name}.pos"] for name in joint_names}

# Print current positions to understand the value range
print("\nCurrent joint positions (normalized -100 to 100):")
for name in joint_names:
    key = f"{name}.pos"
    print(f"  {name:15s}: {q0_dict[key]:8.4f}")

# Create target position - move joint 0 (shoulder_pan) by a noticeable amount
# Since values are normalized between -100 and 100, we need a bigger change
# Let's add 15 units (which should be a noticeable movement)
q1_dict = q0_dict.copy()
first_joint_key = f"{joint_names[0]}.pos"
movement_amount = 15.0  # Move by 15 units in normalized space (about 15% of range)
q1_dict[first_joint_key] += movement_amount

# Clamp to valid range
q1_dict[first_joint_key] = max(-100.0, min(100.0, q1_dict[first_joint_key]))

print(f"\nMoving {joint_names[0]} from {q0_dict[first_joint_key]:.4f} to {q1_dict[first_joint_key]:.4f} (+{movement_amount:.2f})...")
smooth_send(robot, q0_dict, q1_dict)

time.sleep(1.0)  # Wait longer to see the movement

print(f"\nMoving back to original position ({q0_dict[first_joint_key]:.4f})...")
smooth_send(robot, q1_dict, q0_dict)

time.sleep(0.5)

print("\nDone! Movement complete.")
robot.disconnect()
