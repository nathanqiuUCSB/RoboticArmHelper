# main.py
import cv2
import numpy as np
import time
import os
import threading
import queue
from pathlib import Path
import computer_vision
from dotenv import load_dotenv
from gemini import NLPRobotPlanner
from computer_vision import ObjectDetector
from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig
from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.model.kinematics import RobotKinematics
from lerobot.utils.rotation import Rotation
from helper import find_closest_vertical_pixel, STAR_GRAB_POSITIONS, STAR_HOVER_POSITIONS
import helper

load_dotenv()

PORT = os.getenv("ROBOT_PORT", "/dev/tty.usbmodem5A7C1217691")
CALIBRATION_DIR = Path(__file__).parent
ROBOT_ID = "hackathon_robot"

# Starting position - overhead looking down at table (normalized -100 to 100)
# Custom scanning position
STARTING_POSITION = {
    "shoulder_pan.pos": 6.23,
    "shoulder_lift.pos": -35.33,
    "elbow_flex.pos": 17.29,
    "wrist_flex.pos": 86.77,
    "wrist_roll.pos": 52.43,
    "gripper.pos": 49.81,
}

# Global variables for camera display
robot_lock = threading.Lock()  # Lock for robot operations to prevent port conflicts
frame_queue = queue.Queue(maxsize=2)  # Queue to pass frames from background thread to main thread
def take_one_photo(robot,  camera_key="wrist"):
    """Capture a single camera frame from the robot."""
    with robot_lock:
        obs = robot.get_observation()

    if camera_key not in obs:
        raise KeyError(f"Camera '{camera_key}' not found in observation")

    frame = obs[camera_key]

    if not isinstance(frame, np.ndarray):
        raise TypeError("Camera frame is not a numpy array")
    if len(frame.shape) == 3 and frame.shape[2] == 3:
        frame_br = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        return frame_br
                
                # Detect objects
    return frame


def continuous_camera_capture(robot, detector, plan, stop_event):
    """Continuously capture camera frames and process them in background thread."""
    camera_key = "wrist"
    frame_time = 1.0 / 30.0  # 30 fps = 33.33ms per frame
    
    print("Camera capture thread started (30fps)...")
    
    while not stop_event.is_set():
        start_time = time.time()
        
        try:
            # Get camera image with lock to prevent conflicts
            with robot_lock:
                obs = robot.get_observation()
            if camera_key in obs and isinstance(obs[camera_key], np.ndarray):
                frame = obs[camera_key]
                # Convert RGB to BGR for OpenCV if needed
                if len(frame.shape) == 3 and frame.shape[2] == 3:
                    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                    
                    # Detect objects - only detect shape when explicitly looking for it
                    target_color = plan.get('color') if plan else None
                    target_attr = {}
                    if target_color:
                        target_attr['color'] = target_color

                    detections = detector.detect_objects(frame_bgr, target_attribute=target_attr)
                    targets = detector.filter_by_attribute(detections, {'color': target_color} if target_color else {})
                    
                    # Determine target bbox
                    target_bbox = None
                    if targets:
                        if not target_color or 'biggest' in str(plan.get('action', '')).lower():
                            target = detector.get_largest_object(targets)
                            target_bbox = target['bbox']
                        else:
                            target_bbox = targets[0]['bbox']
                    
                    # Draw bounding boxes directly on frame
                    display_frame = frame_bgr.copy()
                    for det in detections:
                        x1, y1, x2, y2 = det['bbox']
                        cls = det.get('class', 'unknown')
                        cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(display_frame, cls, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                    
                    if target_bbox:
                        x1, y1, x2, y2 = target_bbox
                        cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 0, 255), 3)  # Red for target
                    
                    # Put frame in queue for main thread to display (non-blocking)
                    try:
                        frame_queue.put_nowait(display_frame)
                    except queue.Full:
                        # Queue full, skip this frame
                        pass
        except Exception as e:
            # Continue even if there's an error
            pass
        
        # Maintain 30fps
        elapsed = time.time() - start_time
        sleep_time = max(0, frame_time - elapsed)
        if sleep_time > 0:
            time.sleep(sleep_time)
    
    print("Camera capture thread stopped.")


def smooth_move(robot, target_dict, steps=60, dt=0.04):
    """Smoothly move robot to target position (slower for safety)."""
    with robot_lock:
        obs = robot.get_observation()
        joint_names = list(robot.bus.motors.keys())
        start_dict = {f"{name}.pos": obs[f"{name}.pos"] for name in joint_names}
    
    for a in np.linspace(0, 1, steps):
        action = {
            key: (1 - a) * start_dict[key] + a * target_dict[key]
            for key in target_dict.keys()
        }
        with robot_lock:
            robot.send_action(action)
        time.sleep(dt)  # Slower: 0.04s = 25fps movement
        update_camera_display()  # Update display periodically during movement


def move_to_starting_position(robot):
    """Move robot to starting scanning position (overhead looking down)."""
    print("Moving to starting position (overhead)...")
    smooth_move(robot, STARTING_POSITION, steps=60, dt=0.03)
    time.sleep(0.5)
    print("At starting position - camera positioned above table.")

def move_to_star(robot, star_index, shoulder_pan):
    """
    Move to a star position and grab the object there.
    
    Args:
        robot: Robot instance
        star_index: Index of the star (0-13)
    
    Two-stage motion:
    1. Move to "above star" position (hover position)
    2. Move down to "grab" position and close gripper
    """

    CLAW_OFFSET = 7.5

    adjusted_shoulder_pan = shoulder_pan + CLAW_OFFSET
    
    # STAGE 1: Move to hover position above the star
    print(f"Moving to hover position above star {star_index}...")
    hover_position = STAR_HOVER_POSITIONS[star_index]
    hover_position["shoulder_pan.pos"] = adjusted_shoulder_pan
    smooth_move(robot, hover_position, steps=60, dt=0.04)
    time.sleep(0.5)  # Stabilize
    
    # STAGE 2: Move down to grab position
    print(f"Moving down to grab position at star {star_index}...")
    grab_position = STAR_GRAB_POSITIONS[star_index]
    grab_position["shoulder_pan.pos"] = adjusted_shoulder_pan
    smooth_move(robot, grab_position, steps=40, dt=0.04)
    time.sleep(0.5)
    
    # Close gripper to grab object
    print("Closing gripper...")
    close_gripper_position = grab_position.copy()
    close_gripper_position["gripper.pos"] = -50.0  # Closed position (adjust as needed)
    smooth_move(robot, close_gripper_position, steps=20, dt=0.04)
    time.sleep(0.5)
    
    # Optional: Lift object up
    print("Lifting object...")
    lift_position = hover_position.copy()
    lift_position["gripper.pos"] = -50.0  # Keep gripper closed!
    smooth_move(robot, lift_position, steps=40, dt=0.04)
    
    print(f"Successfully grabbed object at star {star_index}!")

def scan_for_object(robot, detector, planner, plan, max_scan_offset=30.0, scan_steps=7):
    """Scan in a grid pattern from starting position to find target object.
    Returns: target dict, best_pan_angle, best_pixel_offset, best_frame, best_scan_pos
    """
    target_name = plan.get('color') or plan.get('shape') or 'target'
    print(f"Scanning for {target_name} object from starting position...")
    
    # Ensure we're at starting position
    move_to_starting_position(robot)
    
    camera_key = "wrist"
    
    # Scan by panning left/right from starting position
    scan_offset_step = (max_scan_offset * 2) / scan_steps
    
    best_target = None
    best_pan_offset = None
    best_pixel_offset = None
    best_frame = None
    best_scan_pos = None
    best_distance_from_center = float('inf')
    
    for i in range(scan_steps):
        pan_offset = -max_scan_offset + (i * scan_offset_step)
        scan_pos = STARTING_POSITION.copy()
        scan_pos["shoulder_pan.pos"] = STARTING_POSITION["shoulder_pan.pos"] + pan_offset
        
        smooth_move(robot, scan_pos, steps=40, dt=0.04)
        time.sleep(0.5)  # Wait for camera to stabilize
        
        # Get camera image and detect
        with robot_lock:
            obs = robot.get_observation()
        if camera_key in obs and isinstance(obs[camera_key], np.ndarray):
            frame = obs[camera_key]
            if len(frame.shape) == 3 and frame.shape[2] == 3:
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                
                # Detect objects
                target_color = plan.get('color')
                target_shape = plan.get("shape")
                target_dict = {}
                if target_color:
                    target_dict['color'] = target_color
                if target_shape:
                    target_dict['shape'] = target_shape
                target_attribute = target_dict if target_dict else None

                detections = detector.detect_objects(frame_bgr, target_attribute=target_attribute)
                targets = detector.filter_by_attribute(detections, target_dict if target_dict else {})
                 
                # Check if target found
                target = None
                if (not target_color and not target_shape) or 'biggest' in str(plan.get('action', '')).lower():
                    if targets:
                        target = detector.get_largest_object(targets)
                elif targets:
                    target = targets[0]
                
                if target:
                    # Calculate pixel offset from image center
                    img_center_x = frame_bgr.shape[1] / 2
                    img_center_y = frame_bgr.shape[0] / 2
                    bbox_center_x = (target['bbox'][0] + target['bbox'][2]) / 2
                    bbox_center_y = (target['bbox'][1] + target['bbox'][3]) / 2
                    pixel_offset = (bbox_center_x - img_center_x, bbox_center_y - img_center_y)
                    
                    # Calculate distance from center (Euclidean distance in pixels)
                    distance_from_center = np.sqrt(pixel_offset[0]**2 + pixel_offset[1]**2)
                    
                    print(f"Found target at pan {pan_offset:.2f}, pixel offset ({pixel_offset[0]:.1f}, {pixel_offset[1]:.1f}), distance from center: {distance_from_center:.1f}px")
                    
                    # Keep track of best view (closest to center)
                    if distance_from_center < best_distance_from_center:
                        best_target = target
                        best_pan_offset = pan_offset
                        best_pixel_offset = pixel_offset
                        best_frame = frame_bgr
                        best_scan_pos = scan_pos.copy()
                        best_distance_from_center = distance_from_center
    
    if best_target is None:
        print("Target object not found during scan.")
        return None, None, None, None, None, None
    
    print(f"\nBest view found at pan angle {best_pan_offset:.2f} with object {best_distance_from_center:.1f}px from center")
    shoulder_pos = best_scan_pos["shoulder_pan.pos"]
    return best_target, best_pan_offset, best_pixel_offset, best_frame, best_scan_pos, shoulder_pos

def update_camera_display():
    """Update camera display from queue - must be called from main thread."""
    window_name = "Robot Camera Feed"
    try:
        # Try to get frame from queue (non-blocking)
        try:
            frame = frame_queue.get_nowait()
            cv2.imshow(window_name, frame)
        except queue.Empty:
            # No new frame, keep showing last frame (or do nothing)
            pass
        cv2.waitKey(1)  # Update window (non-blocking, 1ms timeout)
    except Exception:
        # Window might be closed, ignore
        pass


def main():
    # Initialize components
    print("Initializing robot and components...")
    planner = NLPRobotPlanner()
    detector = ObjectDetector(model_path="yolov8n.pt")
    
    # Configure robot with wrist camera
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
    robot.connect(calibrate=False)
    
    # Create stop event for camera thread
    stop_camera_event = threading.Event()
    camera_thread = None
    
    try:
        
        # Get user instruction
        instruction = input("\nEnter command (e.g., 'pick up the red block' or 'grab the biggest block'): ")
        plan = planner.parse_instruction(instruction)
        print(f"\nPlan: {plan}")

        # Move to starting position (overhead)
        move_to_starting_position(robot)
        update_camera_display()  # Update display in main thread
        
        #Start continuous camera capture thread (background processing)
        camera_thread = threading.Thread(
            target=continuous_camera_capture,
            args=(robot, detector, plan, stop_camera_event),
            daemon=True
        )

        camera_thread.start()
        time.sleep(0.5)  # Give thread time to start
        
        # Scan for object from starting position
        target, best_pan_angle, best_pixel_offset, best_frame, best_scan_pos, shoulder_pos = scan_for_object(robot, detector, planner, plan)
        update_camera_display()  # Update display in main thread
        
        if target is None:
            print("Could not find target object. Exiting.")
            return
        
        # Move back to the position with best view of the object (closest to center)
        print(f"\nMoving back to best view position (pan angle: {best_pan_angle:.2f})...")
        smooth_move(robot, best_scan_pos, steps=60, dt=0.04)
        time.sleep(0.5)  # Wait for camera to stabilize
        update_camera_display()
        print("At best view position - centering object horizontally...")
        

        """
        # Continuously adjust pan to center object horizontally
        target_color = plan.get('color')
        center_tolerance_pixels = 10.0  # Stop when object is within 10 pixels of center horizontally
        max_iterations = 50
        iteration = 0
        
        with robot_lock:
            obs = robot.get_observation()
            joint_names = list(robot.bus.motors.keys())
            current_pos = {f"{name}.pos": obs[f"{name}.pos"] for name in joint_names}
        
        
        while iteration < max_iterations:
            iteration += 1
            update_camera_display()
            time.sleep(0.2)  # Wait a bit for camera to update
            
            # Get current camera frame and detect object
            with robot_lock:
                obs = robot.get_observation()
            camera_key = "wrist"
            if camera_key in obs and isinstance(obs[camera_key], np.ndarray):
                frame = obs[camera_key]
                if len(frame.shape) == 3 and frame.shape[2] == 3:
                    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                    
                    # Detect objects
                    detections = detector.detect_objects(frame_bgr, target_attribute={'color': target_color} if target_color else None)
                    targets = detector.filter_by_attribute(detections, {'color': target_color} if target_color else {})
                    
                    # Find target object
                    target = None
                    if not target_color or 'biggest' in str(plan.get('action', '')).lower():
                        if targets:
                            target = detector.get_largest_object(targets)
                    elif targets:
                        target = targets[0]
                    
                    if target:
                        # Calculate pixel offset from image center
                        img_center_x = frame_bgr.shape[1] / 2
                        bbox_center_x = (target['bbox'][0] + target['bbox'][2]) / 2
                        horizontal_offset_px = bbox_center_x - img_center_x
                        
                        print(f"Iteration {iteration}: Horizontal offset: {horizontal_offset_px:.1f} pixels", end='')
                        
                        # Check if centered (within tolerance)
                        if abs(horizontal_offset_px) < center_tolerance_pixels:
                            print(" ✓ CENTERED!")
                            break
                        
                        # Adjust pan to center object horizontally
                        # Convert pixel offset to pan adjustment (empirical scaling)
                        pan_adjustment = horizontal_offset_px * PIXEL_X_TO_PAN_SCALE
                        
                        # Update current position
                        current_pos["shoulder_pan.pos"] += pan_adjustment
                        current_pos["shoulder_pan.pos"] = max(-100.0, min(100.0, current_pos["shoulder_pan.pos"]))
                        
                        print(f" → Adjusting pan by {pan_adjustment:.2f}")
                        
                        # Move to new position
                        with robot_lock:
                            robot.send_action(current_pos)
                        time.sleep(0.03)
                    else:
                        print(f"Iteration {iteration}: Object not detected, continuing...")
                        time.sleep(0.03)
                        
        
        print("\n" + "="*60)
        print("STAGE 1 COMPLETE - Object centered horizontally in camera frame")
        print("="*60)
        print(f"Final pan position: {current_pos['shoulder_pan.pos']:.2f} (normalized)")
        print("\nProceeding to STAGE 2: Positioning camera directly above object...")
        print("="*60)
        """

        # Take still picture, locate both the colored object AND the X
        target_color = plan.get('color')
        print(f"Target color: {target_color}")
        picture = take_one_photo(robot)

        # Scan for orange
        detections_color = detector.detect_objects(picture, target_attribute={'color': target_color} if target_color else None)
        colored = detector.filter_by_attribute(detections_color, {'color': target_color} if target_color else {})
        print(f"Colored objects found: {colored}")
        color_star_index = helper.find_closest_vertical_pixel(helper.get_center(colored[0]['bbox']))
        print(f"FOUND COLORED OBJECT AT STAR {color_star_index}")

        # Scan for X
        x_plan = {'shape': 'x'}
        x_target, x_best_pan_angle, _, _, x_best_scan_pos, x_shoulder_pos = scan_for_object(robot, detector, planner, x_plan)

        if x_target is not None:
            x_picture = take_one_photo(robot)
            detections_x = detector.detect_objects(x_picture, target_attribute={'shape': 'x'})
            x_objects = detector.filter_by_attribute(detections_x, {'shape': 'x'})
            if x_objects:
                x_star_index = helper.find_closest_vertical_pixel(helper.get_center(x_objects[0]['bbox']))
                print(f"FOUND X AT STAR {x_star_index}")
            else:
                print("WARNING: X not found in confirmation photo")
                x_star_index = None
                x_shoulder_pos = None
        else:
            print("WARNING: X not found during scan - will skip drop step")
            x_star_index = None
            x_shoulder_pos = None

        shoulder_pos = best_scan_pos["shoulder_pan.pos"]

        print(f"Best scan pos: {shoulder_pos}")

        # move to the colored object's star
        move_to_star(robot, color_star_index, shoulder_pos)

        # Now move to the X location and drop the object
        if x_star_index is not None:
            print(f"\nMoving to X location at star {x_star_index}...")
            
            # Move to X's hover position
            x_hover_position = STAR_HOVER_POSITIONS[x_star_index].copy()
            x_hover_position["shoulder_pan.pos"] = x_shoulder_pos + 7.5  # Same offset
            x_hover_position["gripper.pos"] = -50.0  # Keep gripper closed
            smooth_move(robot, x_hover_position, steps=60, dt=0.04)
            time.sleep(0.5)
            
            # Open gripper to drop object
            print("Opening gripper to drop object...")
            x_hover_position["gripper.pos"] = 49.81  # Open gripper
            smooth_move(robot, x_hover_position, steps=20, dt=0.04)
            time.sleep(0.5)
            
            print("Object dropped at X location!")
        else:
            print("Skipping drop - X location not found")

        update_camera_display()  # Update display in main thread
        
        print("\nTask completed!")
        
        # Keep updating display for a bit
        for _ in range(30):  # Update for ~1 second
            update_camera_display()
            time.sleep(0.033)
        
    except KeyboardInterrupt:
        print("\nInterrupted by user")
        stop_camera_event.set()
        if camera_thread:
            camera_thread.join(timeout=1.0)
        cv2.destroyAllWindows()
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        stop_camera_event.set()
        if camera_thread:
            camera_thread.join(timeout=1.0)
        cv2.destroyAllWindows()
    finally:
        # Stop camera display
        stop_camera_event.set()
        if camera_thread:
            camera_thread.join(timeout=1.0)
        
        # Return to starting position (overhead)
        print("\nReturning to starting position (overhead)...")
        move_to_starting_position(robot)
        robot.disconnect()
        print("Robot disconnected.")
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
