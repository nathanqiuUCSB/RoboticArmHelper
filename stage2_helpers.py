"""
Helper functions for STAGE 2: IK-based camera positioning.

Functions to:
1. Convert normalized joint positions to degrees (for IK)
2. Convert degrees back to normalized (for robot control)
3. Create perpendicular camera pose transformation matrix
4. Use IK to position camera close to base, then extend outward
"""

import numpy as np
import json
from pathlib import Path

# Joint limits from URDF (in radians, converted to degrees)
# These are used for converting normalized positions to degrees
JOINT_LIMITS_DEG = {
    "shoulder_pan": (-110.0, 110.0),  # -1.91986 to 1.91986 rad
    "shoulder_lift": (-100.0, 100.0),  # -1.74533 to 1.74533 rad
    "elbow_flex": (-96.8, 96.8),  # -1.69 to 1.69 rad
    "wrist_flex": (-95.0, 95.0),  # -1.65806 to 1.65806 rad
    "wrist_roll": (-157.2, 162.8),  # -2.74385 to 2.84121 rad
    "gripper": (0.0, 100.0)  # Not used in IK, but included for completeness
}


def normalized_to_degrees(normalized_value, joint_name):
    """Convert normalized position (-100 to 100) to degrees.
    
    Args:
        normalized_value: Normalized joint position (-100 to 100)
        joint_name: Name of joint (e.g., "shoulder_pan")
    
    Returns:
        Joint position in degrees
    """
    if joint_name not in JOINT_LIMITS_DEG:
        # Default: assume -100 to 100 maps to -180 to 180 degrees
        return normalized_value * 1.8
    
    min_deg, max_deg = JOINT_LIMITS_DEG[joint_name]
    # Linear mapping: -100 -> min_deg, +100 -> max_deg
    range_deg = max_deg - min_deg
    degrees = ((normalized_value + 100) / 200.0) * range_deg + min_deg
    return degrees


def degrees_to_normalized(degrees_value, joint_name):
    """Convert degrees to normalized position (-100 to 100).
    
    Args:
        degrees_value: Joint position in degrees
        joint_name: Name of joint (e.g., "shoulder_pan")
    
    Returns:
        Normalized joint position (-100 to 100)
    """
    if joint_name not in JOINT_LIMITS_DEG:
        # Default: assume -180 to 180 maps to -100 to 100
        return degrees_value / 1.8
    
    min_deg, max_deg = JOINT_LIMITS_DEG[joint_name]
    # Linear mapping: min_deg -> -100, max_deg -> +100
    range_deg = max_deg - min_deg
    if range_deg == 0:
        return 0.0
    normalized = ((degrees_value - min_deg) / range_deg) * 200.0 - 100.0
    # Clamp to [-100, 100]
    return max(-100.0, min(100.0, normalized))


def joints_normalized_to_degrees(joint_dict, joint_names):
    """Convert dictionary of normalized joint positions to degrees array.
    
    Args:
        joint_dict: Dict with keys like "shoulder_pan.pos" and normalized values
        joint_names: List of joint names (without .pos suffix)
    
    Returns:
        numpy array of joint positions in degrees
    """
    degrees_array = np.zeros(len(joint_names))
    for i, joint_name in enumerate(joint_names):
        key = f"{joint_name}.pos"
        if key in joint_dict:
            degrees_array[i] = normalized_to_degrees(joint_dict[key], joint_name)
    return degrees_array


def joints_degrees_to_normalized(degrees_array, joint_names):
    """Convert array of degrees to dictionary of normalized joint positions.
    
    Args:
        degrees_array: numpy array of joint positions in degrees
        joint_names: List of joint names (without .pos suffix)
    
    Returns:
        Dict with keys like "shoulder_pan.pos" and normalized values
    """
    joint_dict = {}
    for i, joint_name in enumerate(joint_names):
        if i < len(degrees_array):
            normalized = degrees_to_normalized(degrees_array[i], joint_name)
            joint_dict[f"{joint_name}.pos"] = normalized
    return joint_dict


def create_perpendicular_camera_pose(position_xyz, pan_angle_rad):
    """
    Create 4x4 transformation matrix for camera pose perpendicular to ground.
    
    Camera orientation:
    - Z-axis points DOWN (negative world Z) - camera looking straight down
    - X-axis points FORWARD (in direction of pan angle)
    - Y-axis points LEFT (completing right-handed frame)
    
    Args:
        position_xyz: (x, y, z) position in world frame (meters)
        pan_angle_rad: Pan angle in radians (rotation around world Z-axis)
    
    Returns:
        4x4 transformation matrix
    """
    x, y, z = position_xyz
    
    # Camera Z-axis points down (negative world Z)
    camera_z_world = np.array([0.0, 0.0, -1.0])
    
    # Camera X-axis: forward direction based on pan angle
    camera_x_world = np.array([np.cos(pan_angle_rad), np.sin(pan_angle_rad), 0.0])
    
    # Camera Y-axis: left (cross product to complete right-handed frame)
    camera_y_world = np.cross(camera_z_world, camera_x_world)
    camera_y_world = camera_y_world / np.linalg.norm(camera_y_world)
    
    # Re-orthogonalize X (ensure perpendicular)
    camera_x_world = np.cross(camera_y_world, camera_z_world)
    camera_x_world = camera_x_world / np.linalg.norm(camera_x_world)
    
    # Build rotation matrix (columns are camera frame axes in world frame)
    R = np.column_stack([camera_x_world, camera_y_world, camera_z_world])
    
    # Build 4x4 transformation matrix
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = [x, y, z]
    
    return T


def get_pan_angle_rad_from_normalized(pan_normalized):
    """Convert normalized pan position to radians.
    
    Args:
        pan_normalized: Normalized pan position (-100 to 100)
    
    Returns:
        Pan angle in radians
    """
    pan_deg = normalized_to_degrees(pan_normalized, "shoulder_pan")
    return np.radians(pan_deg)
