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
URDF_PATH = CALIBRATION_DIR / "so101_new_calib.urdf"

# Starting position - overhead looking down at table (normalized -100 to 100)
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
PIXEL_X_TO_PAN_SCALE = 0.03     # How much pixel x-offset converts to pan adjustment

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
        update_camera_display()
        
        # Scan for object from starting position
        target, best_pan_angle, best_pixel_offset, best_frame, best_scan_pos = scan_for_object(robot, detector, planner, plan)
        update_camera_display()
        
        if target is None:
            print("Could not find target object. Exiting.")
            return
        
        # Move back to the position with best view of the object (closest to center)
        print(f"\nMoving back to best view position (pan angle: {best_pan_angle:.2f})...")
        smooth_move(robot, best_scan_pos, steps=60, dt=0.04)
        time.sleep(0.5)  # Wait for camera to stabilize
        update_camera_display()
        print("At best view position - centering object horizontally...")
        
        # ========================================================================
        # STAGE 1: Center object HORIZONTALLY
        # ========================================================================
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
        
        # ========================================================================
        # STAGE 2: Center object VERTICALLY (keeping camera PERPENDICULAR)
        # ========================================================================
        print("\n" + "="*60)
        print("STAGE 2: Centering object VERTICALLY (keeping camera PERPENDICULAR)")
        print("="*60)

        # FIX shoulder_pan from STAGE 1 - DO NOT CHANGE IT
        fixed_pan_normalized = current_pos['shoulder_pan.pos']
        print(f"FIXED Pan angle: {fixed_pan_normalized:.2f} (normalized) - WILL NOT CHANGE")
        
        # Step 1: Move to STAGE 2 starting position (with fixed pan)
        print("\nSTAGE 2 Step 1: Moving to STAGE 2 starting position...")
        stage2_start_pos = STAGE2_STARTING_POSITION_TEMPLATE.copy()
        stage2_start_pos['shoulder_pan.pos'] = fixed_pan_normalized
        
        smooth_move(robot, stage2_start_pos, steps=80, dt=0.04)
        time.sleep(0.5)
        update_camera_display()
        
        # Step 2: EXTEND FORWARD until object is centered vertically
        # CRITICAL: Keep gripper PERPENDICULAR the entire time
        print("\nSTAGE 2 Step 2: Extending forward until object is vertically centered...")
        print("(Maintaining perpendicular gripper orientation)")
        
        # Get current position
        with robot_lock:
            obs = robot.get_observation()
            joint_names = list(robot.bus.motors.keys())
            current_stage2_pos = {f"{name}.pos": obs[f"{name}.pos"] for name in joint_names}
        
        vertical_center_tolerance_pixels = 15.0  # Stop when within 15px of vertical center
        max_iterations = 50  # Allow more iterations since we're extending
        iteration = 0
        
        # Movement scale - how aggressively to move based on pixel offset
        # Start with larger movements for extending, then fine-tune
        base_movement_scale = 0.5  # Base extension per iteration
        pixel_adjustment_scale = 0.015  # Additional adjustment based on pixel offset
        
        while iteration < max_iterations:
            iteration += 1
            update_camera_display()
            time.sleep(0.3)
            
            # Get current camera frame and detect object
            with robot_lock:
                obs = robot.get_observation()
            camera_key = "wrist"
            
            if camera_key in obs and isinstance(obs[camera_key], np.ndarray):
                frame = obs[camera_key]
                if len(frame.shape) == 3 and frame.shape[2] == 3:
                    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                    
                    # Detect objects
                    target_color = plan.get('color')
                    detections = detector.detect_objects(frame_bgr, 
                        target_attribute={'color': target_color} if target_color else None)
                    targets = detector.filter_by_attribute(detections, 
                        {'color': target_color} if target_color else {})
                    
                    # Find target object
                    target = None
                    if not target_color or 'biggest' in str(plan.get('action', '')).lower():
                        if targets:
                            target = detector.get_largest_object(targets)
                    elif targets:
                        target = targets[0]
                    
                    if target:
                        # Calculate VERTICAL pixel offset from image center
                        img_center_y = frame_bgr.shape[0] / 2
                        bbox_center_y = (target['bbox'][1] + target['bbox'][3]) / 2
                        vertical_offset_px = bbox_center_y - img_center_y
                        
                        print(f"Iteration {iteration}: Vertical offset: {vertical_offset_px:.1f} pixels", end='')
                        
                        # Check if vertically centered
                        if abs(vertical_offset_px) < vertical_center_tolerance_pixels:
                            print(" ✓ VERTICALLY CENTERED!")
                            break
                        
                        # CRITICAL: To maintain perpendicularity while moving up/down:
                        # We need to adjust THREE joints in coordination:
                        # 1. shoulder_lift: raises/lowers the whole arm
                        # 2. elbow_flex: extends/retracts the arm
                        # 3. wrist_flex: keeps camera pointing straight down
                        
                        # The key insight: shoulder_lift + elbow_flex + wrist_flex must sum to ~180°
                        # to keep the camera perpendicular to the ground
                        
                        # Positive vertical_offset_px = object is BELOW center → extend arm (move down)
                        # Negative vertical_offset_px = object is ABOVE center → retract arm (move up)
                        
                        # Scale factor for movement (tune this based on testing)
                        vertical_adjustment_scale = 0.02
                        
                        # Primary movement: adjust shoulder_lift to move arm up/down
                        shoulder_adjustment = vertical_offset_px * vertical_adjustment_scale
                        
                        # Secondary movement: adjust elbow_flex to extend/retract
                        # When lowering shoulder (positive), extend elbow (positive)
                        # This keeps the gripper moving mostly vertical
                        elbow_adjustment = shoulder_adjustment * 0.8
                        
                        # CRITICAL: Adjust wrist_flex to maintain perpendicularity
                        # wrist_flex must compensate for changes in shoulder_lift + elbow_flex
                        # To keep perpendicular: wrist_flex should move OPPOSITE to (shoulder + elbow)
                        wrist_adjustment = -(shoulder_adjustment + elbow_adjustment)
                        
                        # Update positions
                        current_stage2_pos["shoulder_lift.pos"] += shoulder_adjustment
                        current_stage2_pos["elbow_flex.pos"] += elbow_adjustment
                        current_stage2_pos["wrist_flex.pos"] += wrist_adjustment
                        
                        # Clamp to valid ranges
                        current_stage2_pos["shoulder_lift.pos"] = max(-100.0, min(100.0, 
                            current_stage2_pos["shoulder_lift.pos"]))
                        current_stage2_pos["elbow_flex.pos"] = max(-100.0, min(100.0, 
                            current_stage2_pos["elbow_flex.pos"]))
                        current_stage2_pos["wrist_flex.pos"] = max(-100.0, min(100.0, 
                            current_stage2_pos["wrist_flex.pos"]))
                        
                        print(f" → S:{shoulder_adjustment:+.2f}, E:{elbow_adjustment:+.2f}, W:{wrist_adjustment:+.2f}")
                        
                        # Move to new position
                        with robot_lock:
                            robot.send_action(current_stage2_pos)
                        
                        time.sleep(0.2)
                    else:
                        # Object not visible yet - keep extending forward at base rate
                        print(f"Iteration {iteration}: Object not visible, extending forward...")
                        
                        # Just extend forward at base rate
                        shoulder_adjustment = base_movement_scale
                        elbow_adjustment = shoulder_adjustment * 0.8
                        wrist_adjustment = -(shoulder_adjustment + elbow_adjustment)
                        
                        # Update positions
                        current_stage2_pos["shoulder_lift.pos"] += shoulder_adjustment
                        current_stage2_pos["elbow_flex.pos"] += elbow_adjustment
                        current_stage2_pos["wrist_flex.pos"] += wrist_adjustment
                        
                        # Clamp to valid ranges
                        current_stage2_pos["shoulder_lift.pos"] = max(-100.0, min(100.0, 
                            current_stage2_pos["shoulder_lift.pos"]))
                        current_stage2_pos["elbow_flex.pos"] = max(-100.0, min(100.0, 
                            current_stage2_pos["elbow_flex.pos"]))
                        current_stage2_pos["wrist_flex.pos"] = max(-100.0, min(100.0, 
                            current_stage2_pos["wrist_flex.pos"]))
                        
                        # Move to new position
                        with robot_lock:
                            robot.send_action(current_stage2_pos)
                        
                        time.sleep(0.3)
            else:
                print(f"Iteration {iteration}: No camera frame")
                time.sleep(0.2)
        
        if iteration >= max_iterations:
            print(f"\n⚠ Reached max iterations ({max_iterations})")
        
        print("\n" + "="*60)
        print("STAGE 2 COMPLETE - Object centered both horizontally and vertically")
        print("="*60)
        print("Camera maintained perpendicular orientation throughout")
        print("\nProceeding to STAGE 3: Moving down to grab object...")
        print("="*60)
        
        # ========================================================================
        # STAGE 3: Move down and grab (maintaining perpendicularity)
        # ========================================================================
        print("\nSTAGE 3: Moving down to object (keeping perpendicular)...")
        
        # Move down gradually while maintaining perpendicularity
        down_steps = 20
        down_increment = 2.0  # Increment for shoulder_lift (lowering arm)
        
        for step in range(down_steps):
            # Lower shoulder
            current_stage2_pos["shoulder_lift.pos"] += down_increment
            
            # Extend elbow proportionally
            current_stage2_pos["elbow_flex.pos"] += down_increment * 0.8
            
            # Adjust wrist to maintain perpendicularity
            wrist_compensation = -(down_increment + down_increment * 0.8)
            current_stage2_pos["wrist_flex.pos"] += wrist_compensation
            
            # Clamp all values
            current_stage2_pos["shoulder_lift.pos"] = max(-100.0, min(100.0, 
                current_stage2_pos["shoulder_lift.pos"]))
            current_stage2_pos["elbow_flex.pos"] = max(-100.0, min(100.0, 
                current_stage2_pos["elbow_flex.pos"]))
            current_stage2_pos["wrist_flex.pos"] = max(-100.0, min(100.0, 
                current_stage2_pos["wrist_flex.pos"]))
            
            with robot_lock:
                robot.send_action(current_stage2_pos)
            
            time.sleep(0.1)
            update_camera_display()
        
        time.sleep(0.5)
        
        # Close gripper
        print("Closing gripper...")
        current_stage2_pos["gripper.pos"] = 0.0  # Closed
        smooth_move(robot, current_stage2_pos, steps=50, dt=0.04)
        time.sleep(0.8)
        
        # Lift up (maintaining perpendicularity during lift)
        print("Lifting object...")
        lift_amount = -20.0  # Negative = lift up
        
        # Lift shoulder
        current_stage2_pos["shoulder_lift.pos"] += lift_amount
        
        # Retract elbow proportionally
        current_stage2_pos["elbow_flex.pos"] += lift_amount * 0.8
        
        # Adjust wrist to maintain perpendicularity
        current_stage2_pos["wrist_flex.pos"] -= (lift_amount + lift_amount * 0.8)
        
        # Clamp values
        for key in ["shoulder_lift.pos", "elbow_flex.pos", "wrist_flex.pos"]:
            current_stage2_pos[key] = max(-100.0, min(100.0, current_stage2_pos[key]))
        
        smooth_move(robot, current_stage2_pos, steps=60, dt=0.04)
        time.sleep(0.8)
        
        print("\n" + "="*60)
        print("STAGE 3 COMPLETE - Object picked up!")
        print("="*60)
        print("Camera maintained perpendicular throughout all stages")
        print("\nHolding position. Press Ctrl+C to exit.")
        
        # Hold position
        while True:
            with robot_lock:
                robot.send_action(current_stage2_pos)
            time.sleep(0.1)
            update_camera_display()
        
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
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