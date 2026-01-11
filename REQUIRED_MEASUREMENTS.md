# Required Measurements for Accurate 3D Coordinates

## Current Status
The code currently outputs **approximated 3D coordinates** based on rough estimates. For **accurate calculations**, please provide the following measurements:

---

## 1. ARM LINK LENGTHS (CRITICAL - Most Important)

These are needed for forward kinematics to calculate exact gripper/camera position:

- **Shoulder joint to Elbow joint**: _____ mm or _____ inches
- **Elbow joint to Wrist joint**: _____ mm or _____ inches  
- **Wrist joint to Gripper center**: _____ mm or _____ inches
- **Base height above table/work surface**: _____ mm or _____ inches

**How to measure:**
- Measure center-to-center distances between joint axes
- Base height: Distance from robot base mounting point to table surface

---

## 2. CAMERA PARAMETERS

### Option A: Camera Field of View (FOV)
- **Horizontal FOV**: _____ degrees
- **Vertical FOV**: _____ degrees

**How to find:**
- Check camera specifications/datasheet
- Or calculate: FOV = 2 × arctan(sensor_width / (2 × focal_length))

### Option B: Pixels per inch at scanning height
- **Pixels per inch at table level** (when at starting position): _____ pixels = 1 inch

**How to measure:**
- Place a ruler on the table visible in camera
- Count how many pixels = 1 inch at the scanning height

---

## 3. CAMERA OFFSET FROM GRIPPER

You mentioned camera is 3 inches to the right. Please provide exact 3D offset:

- **X offset (right of gripper center)**: _____ mm or _____ inches (we know ≈ 3 inches)
- **Y offset (forward/back)**: _____ mm or _____ inches
- **Z offset (up/down)**: _____ mm or _____ inches

**Coordinate system relative to gripper center:**
- X: Right is positive
- Y: Forward (away from base) is positive  
- Z: Up is positive

---

## 4. STARTING POSITION HEIGHT

- **Exact gripper center height above table** at starting position: _____ mm or _____ inches
- **OR: Exact camera height above table** at starting position: _____ mm or _____ inches

**How to measure:**
- Measure vertical distance from table surface to gripper center (or camera lens)
- This is critical for accurate Z-coordinate calculation

---

## 5. JOINT ANGLE CONVERSION (Optional - Can be calculated)

If you know the exact relationship between normalized units and degrees:

- **Normalized pan range (-100 to +100)** corresponds to: _____ to _____ degrees
  - Example: -100 to +100 might = -180° to +180°, or -90° to +90°, etc.

**This helps with:**
- Converting pan angles to actual angles for trigonometry
- More accurate X/Y coordinate calculations

---

## Once You Provide These:

1. I'll update the code to use **forward kinematics** to calculate exact camera/gripper position
2. Convert pixel offsets to **real-world distances** using proper camera geometry
3. Calculate **accurate 3D coordinates** using trigonometry and arm geometry
4. Use the coordinates to determine exact joint angles needed (or inverse kinematics if available)

---

## Quick Measurement Guide

### Link Lengths:
1. Put robot in a known configuration (e.g., all joints at 0 or starting position)
2. Measure straight-line distance from:
   - Shoulder pivot axis to elbow pivot axis
   - Elbow pivot axis to wrist pivot axis  
   - Wrist pivot axis to gripper center point
3. Measure from base mounting point straight down to table

### Camera Height:
1. Move robot to starting position
2. Measure vertical distance from table to camera lens (or gripper center)

### Camera FOV:
- Check camera model/specifications online
- Or measure: Place known-size object on table, measure pixels it occupies, calculate FOV

---

**Once you provide these measurements, the 3D coordinate output will be accurate instead of approximated!**
