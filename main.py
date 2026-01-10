# main.py
import cv2
from gemini import NLPRobotPlanner
from computer_vision import ObjectDetector

def main():
    # Initialize
    planner = NLPRobotPlanner()
    detector = ObjectDetector(model_path="yolov8n.pt")
    cap = cv2.VideoCapture(0)  # webcam
    
    # Get user instruction
    instruction = input("Enter command: ")
    plan = planner.parse_instruction(instruction)
    print(f"Plan: {plan}")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Detect objects
        detections = detector.detect_objects(frame, target_attribute={'color': plan['color']})
        targets = detector.filter_by_attribute(detections, {'color': plan['color']})
        
        # Get target and calculate offset
        if targets:
            target = targets[0]
            offset = detector.get_target_offset(target['bbox'], frame.shape)
            print(f"Offset: {offset}")
            
            # TODO: Send offset to robot arm here
            
        # Visualize
        detector.visualize(frame, detections, target['bbox'] if targets else None)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()