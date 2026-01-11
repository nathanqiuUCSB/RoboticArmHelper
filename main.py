# main.py
import cv2
import numpy as np
import time
import os
import threading
import queue
from pathlib import Path
from dotenv import load_dotenv
from gemini import NLPRobotPlanner
from computer_vision import ObjectDetector
from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig
from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig

load_dotenv()

PORT = os.getenv("ROBOT_PORT", "/dev/tty.usbmodem5A7C1217691")
CALIBRATION_DIR = Path(__file__).parent
ROBOT_ID = "hackathon_robot"

# Starting position - overhead looking down at table (normalized -100 to 100)
# Custom scanning position
STARTING_POSITION = {
    "shoulder_pan.pos": 0.0,
    "shoulder_lift.pos": -30.0,
    "elbow_flex.pos": -30.0,
    "wrist_flex.pos": 90.0,
    "wrist_roll.pos": 53.0,
    "gripper.pos": 50.0  # Open gripper
}

# EMPIRICAL TUNING PARAMETERS - Adjust these based on testing results
# These convert pan angle and pixel offsets to joint angle adjustments
PAN_TO_JOINT_SCALE = 1.0        # How much pan (normalized) to adjust shoulder_pan (1:1 ratio)
PIXEL_X_TO_PAN_SCALE = 0.03     # How much pixel x-offset converts to pan adjustment
PIXEL_Y_TO_ELBOW_SCALE = 0.02   # How much pixel y-offset converts to elbow adjustment
CAMERA_OFFSET_COMPENSATION = 10.0  # Pan adjustment to compensate for 3" camera offset (normalized units)
DOWNWARD_MOVEMENT = 25.0        # How much to extend elbow to move down toward object (normalized units)

# Global variables for camera display
robot_lock = threading.Lock()  # Lock for robot operations to prevent port conflicts
frame_queue = queue.Queue(maxsize=2)  # Queue to pass frames from background thread to main thread


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
                    
                    # Detect objects
                    target_color = plan.get('color') if plan else None
                    detections = detector.detect_objects(frame_bgr, target_attribute={'color': target_color} if target_color else None)
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


def scan_for_object(robot, detector, planner, plan, max_scan_offset=30.0, scan_steps=7):
    """Scan in a grid pattern from starting position to find target object.
    Returns: target dict, best_pan_angle, best_pixel_offset, best_frame, best_scan_pos
    """
    print(f"Scanning for {plan.get('color', 'target')} object from starting position...")
    
    # Ensure we're at starting position
    move_to_starting_position(robot)
    
    camera_key = "wrist"
    
    # Scan by panning left/right from starting position
    scan_offset_step = (max_scan_offset * 2) / scan_steps
    
    # Track the best view (object closest to center)
    best_target = None
    best_pan_offset = None
    best_pixel_offset = None
    best_frame = None
    best_scan_pos = None
    best_distance_from_center = float('inf')  # Lower is better (closer to center)
    
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
                detections = detector.detect_objects(frame_bgr, target_attribute={'color': target_color} if target_color else None)
                targets = detector.filter_by_attribute(detections, {'color': target_color} if target_color else {})
                
                # Check if target found
                target = None
                if not target_color or 'biggest' in str(plan.get('action', '')).lower():
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
        return None, None, None, None, None
    
    print(f"\nBest view found at pan angle {best_pan_offset:.2f} with object {best_distance_from_center:.1f}px from center")
    return best_target, best_pan_offset, best_pixel_offset, best_frame, best_scan_pos


# Measurements needed for accurate 3D coordinate calculation:
# ============================================================
# TODO: Measure these to get accurate 3D coordinates:
# 
# 1. ARM LINK LENGTHS (most critical):
#    - Shoulder joint to Elbow joint: ____ mm or inches
#    - Elbow joint to Wrist joint: ____ mm or inches  
#    - Wrist joint to Gripper center: ____ mm or inches
#    - Base height above table: ____ mm or inches
#
# 2. CAMERA PARAMETERS:
#    - Camera Field of View (FOV): Horizontal: ____ degrees, Vertical: ____ degrees
#    OR: Pixels per inch at scanning height: ____ pixels = 1 inch at table level
#    - Camera height above table at starting position: ____ mm or inches
#    - Camera offset from gripper center: X= ____ (3 inches right), Y= ____, Z= ____ mm or inches
#
# 3. STARTING POSITION:
#    - Exact gripper/camera height above table: ____ mm or inches
#
# Once you provide these, we can do proper forward kinematics and accurate 3D calculations!


# MEASURED DIMENSIONS (update these as you get more measurements)
CAMERA_HEIGHT_INCHES = 15.5  # Height of camera at starting position from table top
CAMERA_ANGLE_DEG = -50.0  # Camera angle from flat table top (negative = pointing down)
CAMERA_FOV_HORIZONTAL_DEG = 60.0  # Still need to measure this (or pixels per inch)
CAMERA_OFFSET_RIGHT_INCHES = 3.0  # Camera is 3 inches to the right of gripper center


def calculate_object_3d_coords(pan_angle, pixel_offset, image_shape):
    """Calculate 3D coordinates of object relative to current camera position (origin).
    
    Uses measured camera height and angle for more accurate calculations.
    
    Coordinate system (current camera position as origin 0,0,0):
    - X: Left (-) / Right (+) (positive = right when facing robot)
    - Y: Back (-) / Forward (+) (positive = away from base)
    - Z: Down (-) / Up (+) (positive = up from table, negative = below camera)
    
    Args:
        pan_angle: Pan angle offset from starting position (normalized -100 to 100)
                   At best view position, this is relative to starting position
        pixel_offset: (x, y) pixel offset from image center
        image_shape: (height, width) of image
    
    Returns:
        dict with:
        - 'coords_3d': (x, y, z) coordinates in mm relative to current position
        - 'horizontal_distance_mm': Horizontal distance to object in mm
        - 'vertical_distance_mm': Vertical distance to object in mm (depth)
        - 'angle_deg': Angle to object from camera center
        - 'adjustments': (pan_adjustment, elbow_adjustment, downward_movement) in normalized units
    """
    # MEASURED VALUES
    camera_height_mm = CAMERA_HEIGHT_INCHES * 25.4  # 15.5 inches = 393.7 mm
    camera_angle_rad = np.radians(CAMERA_ANGLE_DEG)  # -50 degrees = pointing down
    
    # Convert pan angle to actual angle
    # Assuming normalized -100 to +100 maps to approximately -180 to +180 degrees
    # This conversion may need adjustment based on actual pan joint range
    pan_angle_deg = pan_angle * 1.8  # Rough conversion (may need calibration)
    pan_angle_rad = np.radians(pan_angle_deg)
    
    # At best view position, we're already aligned, so coordinates are mainly from pixel offset
    # Calculate distance from camera to table center (where camera is pointing)
    camera_to_table_center_dist_mm = camera_height_mm / np.cos(camera_angle_rad)
    
    # Horizontal FOV in mm at table center distance
    fov_horizontal_mm = 2 * camera_to_table_center_dist_mm * np.tan(np.radians(CAMERA_FOV_HORIZONTAL_DEG / 2))
    pixels_per_mm = image_shape[1] / fov_horizontal_mm if fov_horizontal_mm > 0 else 1.0
    
    # Convert pixel offsets to real-world distances on table plane
    # X (left/right): horizontal offset on table
    pixel_x_offset_mm = pixel_offset[0] / pixels_per_mm
    
    # Y (forward/back): need to account for camera angle when projecting onto table
    # When camera is angled, forward/back pixels map to different distances on table
    pixel_y_offset_mm = pixel_offset[1] / pixels_per_mm
    
    # Project to table plane (account for camera angle)
    # X (left/right): mostly unaffected by camera angle for small angles
    x_coord_mm = pixel_x_offset_mm
    
    # Y (forward/back): affected by camera angle
    # Camera points down at angle, so forward pixels correspond to forward on table
    y_coord_mm = pixel_y_offset_mm * np.cos(camera_angle_rad)
    
    # Z coordinate: object is at table level (Z=0), camera is at camera_height_mm above table
    # Origin is at camera position, so table is -camera_height_mm below camera
    z_coord_mm = -camera_height_mm  # Negative = below camera position
    
    # Account for camera offset (3 inches = 76.2 mm to the right)
    # Gripper is 3 inches LEFT of camera, so we need to move gripper RIGHT to align
    camera_offset_mm = CAMERA_OFFSET_RIGHT_INCHES * 25.4  # 3.0 inches = 76.2 mm
    x_coord_mm += camera_offset_mm  # Move gripper right to align with object
    
    # Calculate distances and angles
    # Horizontal distance (in XY plane, perpendicular to camera view)
    horizontal_distance_mm = np.sqrt(x_coord_mm**2 + y_coord_mm**2)
    
    # Vertical distance (depth, along camera view direction)
    # This is the distance from camera to object along the camera's pointing direction
    vertical_distance_mm = np.sqrt(horizontal_distance_mm**2 + abs(z_coord_mm)**2)
    
    # Angle to object from camera center (in degrees)
    # Angle in horizontal plane
    angle_horizontal_deg = np.degrees(np.arctan2(y_coord_mm, x_coord_mm))
    # Angle from camera center (combined)
    angle_to_object_deg = np.degrees(np.arctan2(horizontal_distance_mm, abs(z_coord_mm)))
    
    # Also return joint adjustments (for movement)
    pan_adjustment = (pan_angle * PAN_TO_JOINT_SCALE + 
                     pixel_offset[0] * PIXEL_X_TO_PAN_SCALE + 
                     CAMERA_OFFSET_COMPENSATION)
    elbow_adjustment = pixel_offset[1] * PIXEL_Y_TO_ELBOW_SCALE
    downward_movement = DOWNWARD_MOVEMENT
    
    return {
        'coords_3d': (x_coord_mm, y_coord_mm, z_coord_mm),  # 3D coordinates in mm
        'horizontal_distance_mm': horizontal_distance_mm,
        'vertical_distance_mm': vertical_distance_mm,
        'angle_deg': angle_to_object_deg,
        'angle_horizontal_deg': angle_horizontal_deg,
        'adjustments': (pan_adjustment, elbow_adjustment, downward_movement)
    }


def move_to_object_position(robot, adjustments):
    """Move robot to object position using empirical adjustments.
    
    Args:
        adjustments: (pan_adjustment, elbow_adjustment, downward_movement) in normalized units
    """
    pan_adj, elbow_adj, down_movement = adjustments
    print(f"Moving to object: pan={pan_adj:.2f}, elbow={elbow_adj:.2f}, down={down_movement:.2f}")
    
    # Start from starting position
    move_to_starting_position(robot)
    
    # Calculate target position by adjusting from starting position
    target_pos = STARTING_POSITION.copy()
    
    # Apply adjustments
    target_pos["shoulder_pan.pos"] += pan_adj
    target_pos["elbow_flex.pos"] += elbow_adj + down_movement  # Combine elbow and downward movement
    
    # Clamp values to valid range
    target_pos["shoulder_pan.pos"] = max(-100.0, min(100.0, target_pos["shoulder_pan.pos"]))
    target_pos["elbow_flex.pos"] = max(-100.0, min(100.0, target_pos["elbow_flex.pos"]))
    
    # Move to target position
    smooth_move(robot, target_pos, steps=60, dt=0.04)
    time.sleep(0.5)
    
    return target_pos


def pick_up_object(robot):
    """Move down, close gripper, and lift object."""
    print("Picking up object...")
    
    with robot_lock:
        obs = robot.get_observation()
        joint_names = list(robot.bus.motors.keys())
        current_dict = {f"{name}.pos": obs[f"{name}.pos"] for name in joint_names}
    
    # Move down a bit more
    down_dict = current_dict.copy()
    down_dict["elbow_flex.pos"] += 8.0
    smooth_move(robot, down_dict, steps=40, dt=0.04)
    time.sleep(0.6)
    
    # Close gripper
    grip_dict = down_dict.copy()
    grip_dict["gripper.pos"] = 0.0  # Closed
    smooth_move(robot, grip_dict, steps=50, dt=0.04)
    time.sleep(0.8)
    
    # Lift up
    lift_dict = grip_dict.copy()
    lift_dict["shoulder_lift.pos"] -= 20.0  # Lift arm up
    lift_dict["elbow_flex.pos"] -= 15.0  # Retract elbow
    lift_dict["wrist_flex.pos"] -= 10.0  # Adjust wrist
    
    smooth_move(robot, lift_dict, steps=60, dt=0.04)
    time.sleep(0.8)
    
    print("Object picked up!")


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
        
        # Start continuous camera capture thread (background processing)
        camera_thread = threading.Thread(
            target=continuous_camera_capture,
            args=(robot, detector, plan, stop_camera_event),
            daemon=True
        )
        camera_thread.start()
        time.sleep(0.5)  # Give thread time to start
        
        # Move to starting position (overhead)
        move_to_starting_position(robot)
        update_camera_display()  # Update display in main thread
        
        # Scan for object from starting position
        target, best_pan_angle, best_pixel_offset, best_frame, best_scan_pos = scan_for_object(robot, detector, planner, plan)
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
                        time.sleep(0.1)
                    else:
                        print(f"Iteration {iteration}: Object not detected, stopping.")
                        break
        
        print("\n" + "="*60)
        print("Robot stopped - Object centered horizontally in camera frame")
        print("="*60)
        print(f"Final pan position: {current_pos['shoulder_pan.pos']:.2f} (normalized)")
        print("\nRobot will hold this position. Press Ctrl+C to exit.")
        print("="*60)
        
        # Hold position (keep sending position commands)
        try:
            while True:
                with robot_lock:
                    robot.send_action(current_pos)
                time.sleep(0.1)  # Send position command every 100ms to hold position
                update_camera_display()
        except KeyboardInterrupt:
            print("\nInterrupted by user.")
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
