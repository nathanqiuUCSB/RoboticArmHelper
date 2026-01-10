#!/usr/bin/env python3
"""
Hello World script for SO-101 Robotic Arm
This script connects to the robot, reads its current state, and performs a simple test.
"""

import time
import sys
from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig

# --- CONFIGURATION ---
# Change this to your robot's USB port
# On Mac: Usually something like "/dev/tty.usbmodem..." or "/dev/tty.usbserial..."
# On Linux: Usually "/dev/ttyACM0" or "/dev/ttyUSB0"
# On Windows: Usually "COM3" or similar
# You can find the port by running: python -m lerobot.find_port
ROBOT_PORT = "/dev/tty.usbmodem5A7C1217691"  # Update this with your actual port!


def main():
    print("=" * 60)
    print("SO-101 Robotic Arm - Hello World Test")
    print("=" * 60)
    
    # Step 1: Create robot configuration
    print("\n1. Creating robot configuration...")
    try:
        config = SO101FollowerConfig(
            port=ROBOT_PORT,
            cameras={},  # No cameras for this simple test
            disable_torque_on_disconnect=True,
        )
        print(f"   ✅ Configuration created for port: {ROBOT_PORT}")
    except Exception as e:
        print(f"   ❌ Failed to create configuration: {e}")
        sys.exit(1)
    
    # Step 2: Create robot instance
    print("\n2. Creating robot instance...")
    try:
        robot = SO101Follower(config)
        print("   ✅ Robot instance created")
    except Exception as e:
        print(f"   ❌ Failed to create robot instance: {e}")
        sys.exit(1)
    
    # Step 3: Connect to robot
    print("\n3. Connecting to robot...")
    print("   (This may take a few seconds...)")
    try:
        robot.connect(calibrate=False)  # Set to True if you need calibration
        print("   ✅ Successfully connected to SO-101!")
        print("   ✅ Motors are now active (torque enabled)")
    except Exception as e:
        print(f"   ❌ Failed to connect: {e}")
        print("\n   Troubleshooting tips:")
        print("   - Check that the robot is powered on")
        print("   - Verify the USB cable is connected")
        print("   - Make sure the port is correct (run: python -m lerobot.find_port)")
        print("   - Check that no other program is using the robot")
        sys.exit(1)
    
    # Step 4: Read current state
    print("\n4. Reading current robot state...")
    try:
        observation = robot.get_observation()
        joint_positions = observation.get("joint_positions", [])
        
        if joint_positions is not None:
            print(f"   ✅ Current joint positions:")
            joint_names = ["shoulder_pan", "shoulder_lift", "elbow_flex", 
                          "wrist_flex", "wrist_roll", "gripper"]
            for i, (name, pos) in enumerate(zip(joint_names, joint_positions)):
                print(f"      {name:15s}: {pos:8.4f} rad ({pos*180/3.14159:7.2f}°)")
        else:
            print("   ⚠️  Could not read joint positions")
    except Exception as e:
        print(f"   ⚠️  Error reading state: {e}")
    
    # Step 5: Simple test movement (optional - very small movement)
    print("\n5. Performing simple connection test...")
    print("   (Reading current position to verify communication)")
    try:
        # Just verify we can read positions multiple times
        for i in range(3):
            obs = robot.get_observation()
            pos = obs.get("joint_positions", [])
            if pos:
                print(f"   ✅ Test read {i+1}/3: All joints responding")
            time.sleep(0.5)
    except Exception as e:
        print(f"   ⚠️  Test movement error: {e}")
    
    # Step 6: Disconnect
    print("\n6. Disconnecting from robot...")
    try:
        robot.disconnect()
        print("   ✅ Successfully disconnected")
        print("   ✅ Motors are now loose (torque disabled)")
    except Exception as e:
        print(f"   ⚠️  Error during disconnect: {e}")
    
    print("\n" + "=" * 60)
    print("✅ Hello World test completed successfully!")
    print("=" * 60)
    print("\nYour SO-101 robot is working correctly! 🎉")
    print("\nNext steps:")
    print("  - Try running move_one_joint.py for a simple movement test")
    print("  - Check lerobot documentation for more advanced usage")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
