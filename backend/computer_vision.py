# cv_module.py
import cv2
import numpy as np
from ultralytics import YOLO

class ObjectDetector:
    def __init__(self, model_path=None):
        """
        Initialize YOLOv8 model
        """
        self.model = YOLO(model_path) # Yolo v8
        self.cap = cv2.VideoCapture(2)
        if self.cap.isOpened():
            print(f"Camera works")
    
    def take_picture(self):
        """
        Grab one frame from the camera and return it (BGR image as numpy array).
        """
        ret, frame = self.cap.read()
        if not ret or frame is None:
            raise RuntimeError("Failed to capture image from camera.")
        return frame

    def detect_objects(self, image, target_attribute=None):
        """
        Detect objects in an image
        Returns a list of detections
        Each detection: {'bbox': (x1, y1, x2, y2), 'class': 'red', 'confidence': 0.9}
        """
        detections = []
        results = self.model(image, verbose=False)[0]

        for box, cls, conf in zip(results.boxes.xyxy, results.boxes.cls, results.boxes.conf):
            x1, y1, x2, y2 = map(int, box)
            roi = image[y1:y2, x1:x2]
            hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

            obj_class = 'unknown'

            if target_attribute and "color" in target_attribute and target_attribute["color"]:
                color_name = target_attribute["color"].lower() 
                mask = None

                if color_name == 'red':
                    lower_red1 = np.array([0, 100, 100])
                    upper_red1 = np.array([10, 255, 255])
                    lower_red2 = np.array([160, 100, 100])
                    upper_red2 = np.array([179, 255, 255])
                    mask = cv2.inRange(hsv_roi, lower_red1, upper_red1) + cv2.inRange(hsv_roi, lower_red2, upper_red2)

                elif color_name == 'blue':
                    lower_blue = np.array([90, 40, 80])   # Raised value threshold to 80 for lighter blues
                    upper_blue = np.array([130, 255, 255])  # Tightened hue range to focus on true blues
                    mask = cv2.inRange(hsv_roi, lower_blue, upper_blue)
                
                elif color_name == 'green':
                    lower_green = np.array([40, 50, 50])
                    upper_green = np.array([80, 255, 255])
                    mask = cv2.inRange(hsv_roi, lower_green, upper_green)
                
                elif color_name == 'white':
                    lower_white = np.array([0, 0, 200])
                    upper_white = np.array([179, 30, 255])
                    mask = cv2.inRange(hsv_roi, lower_white, upper_white)

                elif color_name == 'brown':
                    lower_brown = np.array([10, 50, 20])
                    upper_brown = np.array([20, 200, 150])
                    mask = cv2.inRange(hsv_roi, lower_brown, upper_brown)

                elif color_name == 'orange':
                    lower_orange = np.array([5, 50, 50])
                    upper_orange = np.array([30, 255, 255])
                    mask = cv2.inRange(hsv_roi, lower_orange, upper_orange)

                if mask is not None:
                    ratio = np.sum(mask > 0) / mask.size
                    if ratio > 0.3:
                        obj_class = color_name 
                    else:
                        obj_class = "other"

            detections.append({
                'bbox': (x1, y1, x2, y2),
                'class': obj_class,
                'confidence': float(conf)
            })
            
        return detections
    
    def get_largest_object(self, detections):
        """
        Return the detection with the largest bounding box area.
        """
        if not detections:
            return None
        
        largest = max(detections, key=lambda d: (d['bbox'][2] - d['bbox'][0]) * (d['bbox'][3] - d['bbox'][1]))
        return largest

    def filter_by_attribute(self, detections, attribute):
        """
        Filter detections based on attribute from JSON plan
        Example attribute: {'color': 'red'}
        """
        filtered = []
        for det in detections:
            if attribute.get('color') and det.get('class') == attribute.get('color'):
                filtered.append(det)
            elif not attribute.get('color'):
                # If no color specified, include all detections
                filtered.append(det)
        return filtered

    def get_target_offset(self, bbox, image_shape):
        """
        Compute x, y offset from image center
        Returns: {'x_offset': float, 'y_offset': float}
        """
        img_center_x = image_shape[1] / 2
        img_center_y = image_shape[0] / 2

        bbox_center_x = (bbox[0] + bbox[2]) / 2
        bbox_center_y = (bbox[1] + bbox[3]) / 2

        offset = {
            'x_offset': bbox_center_x - img_center_x,
            'y_offset': bbox_center_y - img_center_y
        }
        return offset

    def visualize(self, image, detections, target_bbox=None):
        """
        Optional: draw bounding boxes for debugging/demo
        """
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            cls = det['class']
            cv2.rectangle(image, (x1, y1), (x2, y2), (0,255,0), 2)
            cv2.putText(image, cls, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)

        if target_bbox:
            x1, y1, x2, y2 = target_bbox
            cv2.rectangle(image, (x1, y1), (x2, y2), (0,0,255), 2)  # red for target

        cv2.imshow("Detections", image)
        cv2.waitKey(1)

# Example usage
if __name__ == "__main__":
    detector = ObjectDetector(model_path="path/to/model")
    cap = cv2.VideoCapture(0)  # webcam

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        detections = detector.detect_objects(frame, target_attribute={'color': 'red'})
        red_objects = detector.filter_by_attribute(detections, {'color': 'red'})

        target = red_objects[0] if red_objects else None
        if target:
            offset = detector.get_target_offset(target['bbox'], frame.shape)
            print("Target offset:", offset)

        detector.visualize(frame, detections, target_bbox=target['bbox'] if target else None)