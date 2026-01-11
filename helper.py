STAR_HOVER_POSITIONS = [
    #0
    {
    "shoulder_pan.pos": 4.52,
    "shoulder_lift.pos": -46.82,
    "elbow_flex.pos": 61.23,
    "wrist_flex.pos": 73.62,
    "wrist_roll.pos": 52.43,
    "gripper.pos": 49.48,
    },

    #1
    {
    "shoulder_pan.pos": 8.08,
    "shoulder_lift.pos": -35.63,
    "elbow_flex.pos": 41.09,
    "wrist_flex.pos": 86.30,
    "wrist_roll.pos": 52.38,
    "gripper.pos": 49.55,
    },

    #2,
    {
    "shoulder_pan.pos": 8.52,
    "shoulder_lift.pos": -19.39,
    "elbow_flex.pos": 30.59,
    "wrist_flex.pos": 92.63,
    "wrist_roll.pos": 52.38,
    "gripper.pos": 49.03,
    },

    #3
    {
    "shoulder_pan.pos": 8.82,
    "shoulder_lift.pos": -25.59,
    "elbow_flex.pos": 48.53,
    "wrist_flex.pos": 76.16,
    "wrist_roll.pos": 52.43,
    "gripper.pos": 49.03,
    },

    #4

    {
    "shoulder_pan.pos": 8.67,
    "shoulder_lift.pos": -16.40,
    "elbow_flex.pos": 35.40,
    "wrist_flex.pos": 77.90,
    "wrist_roll.pos": 52.33,
    "gripper.pos": 48.96,
    },

    #5
    {
    "shoulder_pan.pos": 8.30,
    "shoulder_lift.pos": -8.97,
    "elbow_flex.pos": 11.86,
    "wrist_flex.pos": 88.59,
    "wrist_roll.pos": 52.33,
    "gripper.pos": 49.48,
    },

    #6 
    {
    "shoulder_pan.pos": 8.30,
    "shoulder_lift.pos": -5.06,
    "elbow_flex.pos": 12.74,
    "wrist_flex.pos": 86.69,
    "wrist_roll.pos": 52.33,
    "gripper.pos": 49.03,
    },

    #7
    {
    "shoulder_pan.pos": 8.38,
    "shoulder_lift.pos": -4.90,
    "elbow_flex.pos": 16.59,
    "wrist_flex.pos": 77.66,
    "wrist_roll.pos": 52.53,
    "gripper.pos": 49.16,
    },

    #8
    {
    "shoulder_pan.pos": 8.45,
    "shoulder_lift.pos": 4.83,
    "elbow_flex.pos": 11.68,
    "wrist_flex.pos": 77.35,
    "wrist_roll.pos": 52.67,
    "gripper.pos": 48.96,
    },

    #9
    {
    "shoulder_pan.pos": 8.15,
    "shoulder_lift.pos": 17.62,
    "elbow_flex.pos": -10.46,
    "wrist_flex.pos": 77.50,
    "wrist_roll.pos": 52.77,
    "gripper.pos": 48.96,
    },

    #10

]

STAR_GRAB_POSITIONS = [
    #0
    {
    "shoulder_pan.pos": 6.82,
    "shoulder_lift.pos": -67.05,
    "elbow_flex.pos": 93.35,
    "wrist_flex.pos": 60.48,
    "wrist_roll.pos": 52.48,
    "gripper.pos": 54.74,
    },

    #1
     {
    "shoulder_pan.pos": 12.90,
    "shoulder_lift.pos": -15.25,
    "elbow_flex.pos": 63.15,
    "wrist_flex.pos": 57.62,
    "wrist_roll.pos": 52.43,
    "gripper.pos": 45.91,
    },

    #2
    {
    "shoulder_pan.pos": 8.75,
    "shoulder_lift.pos": -6.05,
    "elbow_flex.pos": 51.77,
    "wrist_flex.pos": 65.19,
    "wrist_roll.pos": 52.48,
    "gripper.pos": 48.96,
    },

    #3,
    {
    "shoulder_pan.pos": 8.82,
    "shoulder_lift.pos": -14.48,
    "elbow_flex.pos": 61.31,
    "wrist_flex.pos": 55.88,
    "wrist_roll.pos": 52.67,
    "gripper.pos": 48.96,
    },

    #4
    {
    "shoulder_pan.pos": 8.82,
    "shoulder_lift.pos": -4.98,
    "elbow_flex.pos": 49.76,
    "wrist_flex.pos": 55.96,
    "wrist_roll.pos": 52.38,
    "gripper.pos": 49.03,
    },

    #5
    {
    "shoulder_pan.pos": 8.45,
    "shoulder_lift.pos": -4.67,
    "elbow_flex.pos": 46.00,
    "wrist_flex.pos": 53.90,
    "wrist_roll.pos": 52.43,
    "gripper.pos": 45.45,
    },

    #6
    {
    "shoulder_pan.pos": 8.30,
    "shoulder_lift.pos": 7.59,
    "elbow_flex.pos": 31.82,
    "wrist_flex.pos": 56.83,
    "wrist_roll.pos": 52.58,
    "gripper.pos": 49.16,
    },

    #7
    {
    "shoulder_pan.pos": 8.30,
    "shoulder_lift.pos": 1.23,
    "elbow_flex.pos": 35.93,
    "wrist_flex.pos": 52.87,
    "wrist_roll.pos": 52.53,
    "gripper.pos": 49.09,
    },

    #8
    {
    "shoulder_pan.pos": 8.38,
    "shoulder_lift.pos": 11.49,
    "elbow_flex.pos": 23.24,
    "wrist_flex.pos": 58.97,
    "wrist_roll.pos": 52.58,
    "gripper.pos": 48.96,
    },

    #9
    {
    "shoulder_pan.pos": 8.15,
    "shoulder_lift.pos": 22.84,
    "elbow_flex.pos": 2.67,
    "wrist_flex.pos": 63.33,
    "wrist_roll.pos": 52.63,
    "gripper.pos": 48.96,
    },

    #10

]
def get_center(bbox):
    x1, y1, x2, y2 = bbox   
    center_y = (y1 + y2) / 2
    
    return center_y
def find_closest_vertical_pixel(targetpixel: int):
    y_sections = [409, 379, 350, 319, 287, 257, 225, 191, 160, 129]
    bottombound = 424
    
    if targetpixel > bottombound:
        return -1
    
    closest_index = 0
    min_distance = abs(targetpixel - y_sections[0])
    
    for i in range(1, len(y_sections)):
        distance = abs(targetpixel - y_sections[i])
        if distance < min_distance:
            min_distance = distance
            closest_index = i
    
    return closest_index
    
    
