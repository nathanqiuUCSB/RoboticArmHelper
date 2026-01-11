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
from lerobot.model.kinematics import RobotKinematics
from lerobot.utils.rotation import Rotation

load_dotenv()

PORT = os.getenv("ROBOT_PORT", "/dev/tty.usbmodem5A7C1217691")
CALIBRATION_DIR = Path(__file__).parent
ROBOT_ID = "hackathon_robot"
URDF_PATH = CALIBRATION_DIR / "so101_new_calib.urdf"

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

# STAGE 2 Starting Position Template (with shoulder_pan.pos = 0.0 as template)
# The actual shoulder_pan.pos from STAGE 1 will replace the 0.0 value
STAGE2_STARTING_POSITION_TEMPLATE = {
    "shoulder_pan.pos": 0.0,
    "shoulder_lift.pos": -78.47,
    "elbow_flex.pos": 66.13,
    "wrist_flex.pos": 91.60,
    "wrist_roll.pos": 54.68,
    "gripper.pos": 49.42,
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
                        print(f"Iteration {iteration}: Object not detected, continuing...")
                        time.sleep(0.1)
        
        print("\n" + "="*60)
        print("STAGE 1 COMPLETE - Object centered horizontally in camera frame")
        print("="*60)
        print(f"Final pan position: {current_pos['shoulder_pan.pos']:.2f} (normalized)")
        print("\nProceeding to STAGE 2: Positioning camera directly above object...")
        print("="*60)
        
        # STAGE 2: Position camera directly above object, perpendicular to ground
        try:
            from stage2_helpers import (
                joints_normalized_to_degrees,
                joints_degrees_to_normalized,
                create_perpendicular_camera_pose,
                get_pan_angle_rad_from_normalized
            )
            
            # Initialize kinematics solver
            print("\nInitializing kinematics solver...")
            try:
                kinematics = RobotKinematics(
                    urdf_path=str(URDF_PATH),
                    target_frame_name="gripper_frame_link",
                    joint_names=joint_names,
                )
                print("Kinematics solver initialized.")
            except ImportError as e:
                print(f"Error initializing kinematics: {e}")
                raise
            
            # FIX shoulder_pan position from STAGE 1 - DO NOT CHANGE IT
            fixed_pan_normalized = current_pos['shoulder_pan.pos']
            fixed_pan_deg = joints_normalized_to_degrees({'shoulder_pan.pos': fixed_pan_normalized}, ['shoulder_pan'])[0]
            pan_angle_rad = np.radians(fixed_pan_deg)
            print(f"FIXED Pan angle: {fixed_pan_deg:.2f} degrees (normalized: {fixed_pan_normalized:.2f}) - WILL NOT CHANGE")
            
            # Find shoulder_pan index in joint_names
            pan_index = joint_names.index('shoulder_pan') if 'shoulder_pan' in joint_names else None
            
            # Step 1: Create STAGE 2 starting position from template
            # Copy the template and replace shoulder_pan.pos with the value from STAGE 1
            print("\nSTAGE 2 Step 1: Moving to STAGE 2 starting position (template with fixed pan)...")
            stage2_start_pos = STAGE2_STARTING_POSITION_TEMPLATE.copy()
            stage2_start_pos['shoulder_pan.pos'] = fixed_pan_normalized
            print(f"  Using template position with shoulder_pan.pos = {fixed_pan_normalized:.2f}")
            
            # Move to STAGE 2 starting position
            print("Moving to STAGE 2 starting position...")
            smooth_move(robot, stage2_start_pos, steps=80, dt=0.04)
            time.sleep(0.5)
            update_camera_display()
            
            # Get current joint positions in degrees for IK calculations
            with robot_lock:
                obs = robot.get_observation()
            current_stage2_pos = {f"{name}.pos": obs[f"{name}.pos"] for name in joint_names}
            current_joints_deg = joints_normalized_to_degrees(current_stage2_pos, joint_names)
            
            # FIX wrist_roll from STAGE 2 starting position - DO NOT CHANGE IT (keep camera orientation stable)
            fixed_wrist_roll_normalized = current_stage2_pos['wrist_roll.pos']
            fixed_wrist_roll_deg = joints_normalized_to_degrees({'wrist_roll.pos': fixed_wrist_roll_normalized}, ['wrist_roll'])[0]
            print(f"FIXED Wrist roll: {fixed_wrist_roll_deg:.2f} degrees (normalized: {fixed_wrist_roll_normalized:.2f}) - WILL NOT CHANGE")
            
            # Find joint indices
            wrist_roll_index = joint_names.index('wrist_roll') if 'wrist_roll' in joint_names else None
            
            # Get initial camera pose using forward kinematics
            initial_pose = kinematics.forward_kinematics(current_joints_deg)
            initial_pos_xyz = initial_pose[:3, 3]
            initial_distance_xy = np.sqrt(initial_pos_xyz[0]**2 + initial_pos_xyz[1]**2)
            fixed_z_height = initial_pos_xyz[2]  # CRITICAL: Keep this Z height constant (parallel to floor)
            
            print(f"Initial camera position: ({initial_pos_xyz[0]:.3f}, {initial_pos_xyz[1]:.3f}, {initial_pos_xyz[2]:.3f}) m")
            print(f"Initial distance from base (XY): {initial_distance_xy:.3f} m")
            print(f"FIXED Z height: {fixed_z_height:.3f} m - WILL REMAIN CONSTANT (parallel to floor)")
            
            # Step 2: Extend outward while keeping camera perpendicular and at constant height
            print("\nSTAGE 2 Step 2: Extending forward while keeping camera perpendicular and at constant height...")
            extension_step = 0.02  # 2 cm steps
            max_extension_distance = 0.30  # Extend up to 30 cm from initial position
            max_iterations = int(max_extension_distance / extension_step)  # Calculate max iterations
            
            iteration = 0
            
            while iteration < max_iterations:
                iteration += 1
                update_camera_display()
                time.sleep(0.2)
                
                # Get current pose
                current_pose = kinematics.forward_kinematics(current_joints_deg)
                current_pos_xyz = current_pose[:3, 3]
                current_distance_xy = np.sqrt(current_pos_xyz[0]**2 + current_pos_xyz[1]**2)
                
                # Calculate target distance (extend outward)
                target_distance = initial_distance_xy + (iteration * extension_step)
                
                # Stop if we've reached max extension
                if target_distance > initial_distance_xy + max_extension_distance:
                    print(f"\n✓ STAGE 2 COMPLETE! Reached maximum extension distance ({max_extension_distance:.3f} m)")
                    break
                
                target_x = target_distance * np.cos(pan_angle_rad)
                target_y = target_distance * np.sin(pan_angle_rad)
                target_z = fixed_z_height  # CRITICAL: Keep Z constant (parallel to floor)
                
                # Create perpendicular pose
                target_pose = create_perpendicular_camera_pose((target_x, target_y, target_z), pan_angle_rad)
                
                # Solve IK
                try:
                    target_joints_deg = kinematics.inverse_kinematics(
                        current_joints_deg,
                        target_pose,
                        position_weight=1.0,
                        orientation_weight=1.0  # High weight to keep perpendicular
                    )
                    # CRITICAL: Lock shoulder_pan and wrist_roll to fixed values - DO NOT CHANGE THEM
                    # Let IK solve for elbow_flex, shoulder_lift, and wrist_flex to maintain constant Z height
                    if pan_index is not None:
                        target_joints_deg[pan_index] = fixed_pan_deg
                    if wrist_roll_index is not None:
                        target_joints_deg[wrist_roll_index] = fixed_wrist_roll_deg
                    current_joints_deg = target_joints_deg
                except Exception as e:
                    print(f"IK solve failed at iteration {iteration}: {e}")
                    print(f"  Current distance: {current_distance_xy:.3f}m, Target distance: {target_distance:.3f}m")
                    break
                
                # Convert to normalized and move
                target_joints_normalized = joints_degrees_to_normalized(target_joints_deg, joint_names)
                # Ensure locked joints are exactly fixed in normalized units too
                target_joints_normalized['shoulder_pan.pos'] = fixed_pan_normalized
                target_joints_normalized['wrist_roll.pos'] = fixed_wrist_roll_normalized
                
                with robot_lock:
                    robot.send_action(target_joints_normalized)
                
                # Verify actual Z height after move
                verify_pose = kinematics.forward_kinematics(current_joints_deg)
                verify_z = verify_pose[2, 3]
                print(f"Iteration {iteration}: Distance={target_distance:.3f}m, Z={verify_z:.3f}m (target: {fixed_z_height:.3f}m)")
                
                time.sleep(0.3)
            
            if iteration >= max_iterations:
                print(f"\n✓ STAGE 2 COMPLETE! Extended for {max_iterations} iterations ({max_extension_distance:.3f} m)")
            
            # Get final position for display
            final_pose = kinematics.forward_kinematics(current_joints_deg)
            final_pos_xyz = final_pose[:3, 3]
            final_distance_xy = np.sqrt(final_pos_xyz[0]**2 + final_pos_xyz[1]**2)
            
            print("\n" + "="*60)
            print("STAGE 2 COMPLETE - Extended forward while keeping camera perpendicular")
            print("="*60)
            final_joints_normalized = joints_degrees_to_normalized(current_joints_deg, joint_names)
            print(f"Final distance from base: {final_distance_xy:.3f} m")
            print(f"Final Z height: {final_pos_xyz[2]:.3f} m (initial: {fixed_z_height:.3f} m)")
            print("\nRobot will hold this position. Press Ctrl+C to exit.")
            print("="*60)
            
            # Hold position
            while True:
                with robot_lock:
                    robot.send_action(final_joints_normalized)
                time.sleep(0.1)
                update_camera_display()
                
        except Exception as e:
            error_msg = str(e)
            if "placo" in error_msg.lower():
                print(f"\nError: {error_msg}")
                print("STAGE 2 requires placo library. Please install it with:")
                print("  pip install 'placo>=0.9.6,<0.10.0'")
            else:
                print(f"\nError in STAGE 2: {error_msg}")
                import traceback
                traceback.print_exc()
            print("STAGE 2 skipped - falling back to holding STAGE 1 position")
            while True:
                with robot_lock:
                    robot.send_action(current_pos)
                time.sleep(0.1)
                update_camera_display()
        except KeyboardInterrupt:
            print("\nInterrupted by user.")
        except Exception as e:
            print(f"\nError in STAGE 2: {e}")
            import traceback
            traceback.print_exc()
            print("Falling back to holding STAGE 1 position")
            while True:
                with robot_lock:
                    robot.send_action(current_pos)
                time.sleep(0.1)
                update_camera_display()
        
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
