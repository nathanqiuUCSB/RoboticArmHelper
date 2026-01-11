# Required Dimensions for Accurate 3D Coordinate Calculation

To properly calculate 3D coordinates and position the robot accurately, we need:

## Critical Dimensions Needed:

### 1. **Robot Arm Link Lengths** (most important)
   - **Shoulder to Elbow length**: Distance from shoulder joint to elbow joint (in mm or inches)
   - **Elbow to Wrist length**: Distance from elbow joint to wrist joint (in mm or inches)  
   - **Wrist to Gripper/Camera length**: Distance from wrist joint to gripper center (in mm or inches)
   - **Base height**: Height of robot base above table/work surface (in mm or inches)

### 2. **Camera Parameters**
   - **Camera Field of View (FOV)**: Horizontal and vertical FOV in degrees
   - OR: **Pixels per inch at table distance**: How many pixels = 1 inch at the scanning height
   - **Camera height at starting position**: Exact height of camera above table (in mm or inches)
   - **Camera offset from gripper**: We know it's 3 inches to the right, but need exact 3D offset (x, y, z)

### 3. **Starting Position Information**
   - **Exact height of gripper/camera above table** when at starting position
   - This helps convert pixel offsets to real-world distances

### 4. **Joint Angle to Real-World Conversion**
   - How much horizontal distance does 1 normalized pan unit correspond to at table level?
   - This depends on the arm configuration and heights

## Alternative: Use Kinematics Functions

If lerobot has kinematics functions for SO-101, we could:
- Use **forward kinematics** to calculate exact gripper position at starting position
- Use **inverse kinematics** to calculate joint angles needed to reach target 3D position
- This would be more accurate than manual calculations

## Current Approach (Using Approximations)

Right now, the code uses rough approximations:
- Assumes ~0.12 inches per normalized pan unit
- Uses pixel-to-distance conversion estimates
- These approximations are causing the gripper to be too high

If you can provide the link lengths or we can use kinematics, we can make this much more accurate!
