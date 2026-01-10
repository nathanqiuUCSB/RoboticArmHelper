import time
import sys
from lerobot.common.robot_devices.robots.so100 import SO100Robot

# --- CONFIGURATION ---
# CHANGE THIS to your specific port!
# Windows example: "COM3"
# Linux/Mac example: "/dev/ttyACM0" or "/dev/tty.usbmodem..."
ROBOT_PORT = "/dev/tty.usbmodem5A7C1217691" 

def main():
    print(f"1. Attempting to connect to {ROBOT_PORT}...")
    
    # Configure the robot connection
    try:
        config = SO100Robot.config_class(port=ROBOT_PORT, cameras={})
        robot = SO100Robot(config)
        robot.connect()
        print("   ✅ Connected! Motors are now stiff (Torque ON).")
    except Exception as e:
        print(f"   ❌ Failed to connect: {e}")
        print("      Check your USB cable and PORT variable.")
        sys.exit(1)

    # Read current position (Safety first!)
    # SO-100 Joints: [Base, Shoulder, Elbow, Wrist-Pitch, Wrist-Roll, Gripper]
    start_pos = robot.read("present_position")
    print(f"2. Current Positions: {start_pos.tolist()}")

    # Define 'Up' and 'Down' relative to where we are now
    # We will wiggle Joint 1 (The Shoulder) and Joint 2 (The Elbow)
    # NOTE: We use .clone() so we don't mess up the original data
    pos_down = start_pos.clone()
    pos_up = start_pos.clone()

    # Create a gentle nodding motion
    # Adjust these values (radians) if you want bigger movement
    pos_up[1] -= 0.3   # Move shoulder back/up
    pos_up[2] -= 0.3   # Move elbow back/up
    
    pos_down[1] += 0.2 # Move shoulder forward/down
    pos_down[2] += 0.2 # Move elbow forward/down

    print("3. Starting movement loop (Press Ctrl+C to Stop)...")
    
    try:
        for i in range(3): # Run 3 times
            print(f"   Loop {i+1}/3: Moving UP...")
            robot.write("goal_position", pos_up)
            time.sleep(2.0) # Wait for move to finish

            print(f"   Loop {i+1}/3: Moving DOWN...")
            robot.write("goal_position", pos_down)
            time.sleep(2.0)

    except KeyboardInterrupt:
        print("\n   ! Emergency Stop triggered by user.")

    finally:
        # cleanup
        print("4. Disconnecting...")
        robot.disconnect()
        print("   ✅ Disconnected. Motors should be loose.")

if __name__ == "__main__":
    main()