# SO-101 Robotic Arm - Vision-Based Pick and Place

This system allows a robotic arm to pick up objects using vision-based detection with natural language commands.

## Prerequisites

1. **Python Environment**: Python 3.10+ with required packages
2. **Robot Connection**: SO-101 robot connected via USB
3. **Camera**: Wrist-mounted camera connected
4. **Environment Variables**: `.env` file with required keys

## Setup

### 1. Install Dependencies

Make sure you have all required packages installed:
```bash
pip install -r requirements.txt
```

Additional packages needed:
- `ultralytics` (for YOLOv8)
- `groq` (for NLP planning)
- `opencv-python` or `opencv-python-headless`
- `python-dotenv`

### 2. Set Up Environment Variables

Create a `.env` file in the project directory with:
```env
ROBOT_PORT=/dev/tty.usbmodem5A7C1217691
GROQ_API_KEY=your_groq_api_key_here
```

**Find your robot port:**
```bash
python -m lerobot.find_port
```

**Get a Groq API key:**
- Sign up at https://console.groq.com/
- Create an API key
- Add it to your `.env` file

### 3. Download YOLOv8 Model

The script uses `yolov8n.pt`. If it's not in the directory, it will download automatically, or you can download manually.

## Running the Script

### Basic Usage

```bash
python main.py
```

### What to Expect

1. **Initialization**: The script will:
   - Initialize the NLP planner (Gemini/Groq)
   - Load the object detector (YOLOv8)
   - Connect to the robot
   - Connect to the wrist camera

2. **User Input**: You'll be prompted to enter a command:
   ```
   Enter command (e.g., 'pick up the red block' or 'grab the biggest block'):
   ```

3. **Execution**: The robot will:
   - Move to starting position
   - Scan left/right to find the target object
   - Approach the detected object
   - Pick it up with the gripper
   - Return to starting position

### Example Commands

- `"pick up the red block"`
- `"grab the blue block"`
- `"pick up the green block"`
- `"grab the biggest block"`

## Troubleshooting

### Camera Issues

If the camera doesn't work, try adjusting the camera index:
```python
# In main.py, line 175, try different indices:
"wrist": OpenCVCameraConfig(index_or_path=0, ...)  # Try 0, 1, 2, etc.
```

List available cameras:
```python
import cv2
for i in range(5):
    cap = cv2.VideoCapture(i)
    if cap.isOpened():
        print(f"Camera {i} is available")
        cap.release()
```

### Robot Connection Issues

- Make sure the robot is powered on
- Check USB cable connection
- Verify the port in `.env` matches your robot
- Ensure no other program is using the robot

### Calibration

The script uses `hackathon_robot.json` for calibration. Make sure it's in the same directory as `main.py`.

### Starting Position

If the starting position needs adjustment, edit `STARTING_POSITION` in `main.py` (around line 20). Values are normalized between -100 and 100.

## File Structure

```
RoboticArmHelper/
├── main.py                 # Main script
├── gemini.py              # NLP planner
├── computer_vision.py     # Object detector
├── hackathon_robot.json   # Robot calibration
├── yolov8n.pt            # YOLOv8 model
├── .env                  # Environment variables (create this)
└── requirements.txt      # Python dependencies
```

## Safety Notes

- Always ensure the robot workspace is clear
- Be ready to stop the script (Ctrl+C) if needed
- Check that the gripper is open before starting
- Verify the starting position is safe for your setup
