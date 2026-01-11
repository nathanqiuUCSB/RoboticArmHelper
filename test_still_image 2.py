import helper
from computer_vision import ObjectDetector, take_picture

detector = ObjectDetector(model_path="yolov8n.pt")
#picture =
boundary_box = detector.detect_objects(picture, target_attribute={'color':"red"} if "red" else None)
vertical_group = helper.findClosestVerticalPixel(helper.get_center(boundary_box))
print(vertical_group)