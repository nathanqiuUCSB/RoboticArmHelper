# SB HACKS WINNER: ExtendAble - AI Robotic Arm Project

A voice-controlled robotic arm that uses computer vision, LLMs, and feedback-based control to locate, align with, and grasp objects from natural language commands.

Designed for SB HACKS XII (2026) by Ahmed Alhakem, Beckett Hayes, Nathan Qiu, and Joshua Gray. 


Check out our Devpost at: https://devpost.com/software/robotic-arm-j4fmbq

Check out our Youtube Video at: https://www.youtube.com/watch?v=e0lhDRYjyEc


## Inspiration

Our inspiration for this project came from our fascination with the idea of building an autonomous robot of some sort that could perform human-like tasks. We were particularly inspired by the idea of robotics supplementing human capability by providing functionality that could assist people with every day tasks. The concepts of autonomous robotic arms could be specifically applied to people with disabilities by providing them new capabilities and expanding their physical abilities. This is a growing industry that interests us greatly. 


## What it does

The robotic arm takes in a voice command telling it to grab an object (Eg: "Grab an orange"). After this input, the robotic arm will automatically locate the desired object, position itself to grab it, and carry it onto a "target pad" which is a blue X.


## How to Run / Try It Out

To run the full system locally, you must start both the backend API (robot + vision + LLM pipeline) and the speech-to-text frontend.

### 1. Start the Backend (Robot + Vision + LLM)

From the root of the project:

```bash
cd backend
.../miniforge3/envs/lerobot/bin/python -m uvicorn main:app --reload
```

This launches the FastAPI server that:

	• Handles speech-to-text processing
	
	• Communicates with the LLM

	• Runs computer vision and robot control logic


⚠️ Note: This assumes the lerobot conda environment is installed at the path above.

⸻

### 2. Start the Speech-to-Text Frontend

In a separate terminal window:

```bash
cd stt-frontend
npm install   # first time only
npm run dev
```
This launches the frontend interface used to record voice commands and send them to the backend.

⸻

### 3. Using the Robot

1. Start both servers


2. Speak a command such as:

	• “Grab an orange”

	• “Pick up the red object”


3. The robotic arm will:

	• Locate the object

	• Align itself

	• Pick it up

	• Place it on the blue target pad (X)


## System Architecture

Voice Command
→ Speech-to-Text (Frontend)
→ FastAPI Backend
→ LLM Parsing (Object + Color)
→ Computer Vision (YOLO + Camera)
→ Partition-Based Motion Planning
→ Servo Control (SO-101 Arm)
→ Target Placement


## How we built it

The physical robot was an SO-101 ARM, whose pieces were 3d printed based on designs from an open source repository. 

First, the robot receives instructions regarding what object it will pick up. It does this through speech-to-text extraction using the Python Library speechRecongition.  After we obtained the text of the request, we used a Groq LLM to parse the text and accurately extract distinguishing characteristics of the target that our robot arm could use to locate. In our case, we focused on the color of the target. For example, if the user asks to grab an apple, we determine that we're looking for a red object. 

After extracting information regarding the color of the target, we needed the arm to search for it with its built-in camera. The camera uses YOLO to analyze its environment. First, the arm scans its surroundings by rotating, keeping note of what position in which the object was in the middle of horizontal of the screen, as well as what position the target pad was in. After scanning its environment and calculating the optimal rotation of the robot, the arm looks to pick up the desired object.

Once we have fixed the rotation, we have restricted a dimension and thus reduced the problem to a 2-dimensional plane. Using computer vision, we partitioned the environment into 10 discrete zones, with increasing distance away from the robotic arm. We wrote an algorithm to analyze the image of the field and determine what partition the object was in. 

After determining the partition of the object, we needed to find a way to move to it. We developed a function that would move the servos and motors smoothly and cohesively into a new configuration. Then, we manually programmed a sequence of points that represented the shape of the trajectory of the robot arm for each partition. 

This meant that once we knew what partition the object was in, and we aligned it horizontally in the center, we were able to determine the sequence of points to move in and successfully grab the object.

After grabbing the object, we used the same procedure to position the arm onto the target pad and release the object.


## Challenges we ran into

There were two main challenges that we faced throughout the project. The first was developing a method to locate the objects in 3d space, and the second is accurately controlling the robot arm to move the head onto the object and pick it up. This was a major challenge due to the fact that we had only one camera, which made it difficult to interpret the 3d space of the robot. We attempted to use some Inverse Kinematic Libraries, but we found they were always slightly off and unable to perform in an acceptable quality for us. 

Additionally, the robot design and libraries we used were oriented around creating agentic robotic arms, which we didn't have the time or computation power to train. This meant that instead of relying on abstracted movements, we had to manually program the individual motors themselves to work together. 


## Accomplishments that we're proud of

We're most proud of developing our own custom system that enabled the robot to visualize objects in a 3d space and accurately move toward them. As mentioned before, we developed a strategy where we first rotate the arm in order to align the object horizontally within the frame, and afterwards we identify what partition the object is in relative to the camera and follow a pre-computed trajectory toward the goal state. 

We're particularly proud of this approach for our creative problem solving and thinking outside the box, as we were able to find an alternate solution that differed from the existing ones provided in the form of Inverse Kinematic Libraries and the Agentic Frameworks of the robotic arm. We were also able to successfully and accurately move the robot arm with 6 degrees of freedom in a 3 dimensional space with only one arm and no automatic library abstractions. 

On a personal level, this also symbolized our resilience and perseverance, as it was around 3AM when we thought of this new idea and pivoted away from our previous efforts of using inverse kinematics after being stuck for hours. 


## What we learned

This was our first experience working with robotics and embedded systems, so we learned a vast amount of what it entails to design software that drives hardware. Additionally, we learned how to debug embedded firmware in the context of writing tests for hardware, and making small incremental goals, such as simply moving the robot up and down as a "hello world". 

We also gained new insights into computer vision and how we can use YOLO, image detection, and mapping from a 2d to 3d world. 

We also significantly improved our prompt engineering skills and had lots of practice detailing precise, specific instructions to LLMs to help resolve issues, debug glitches, and generate tedious code.


## What's next for Robotic Arm

Next, we would love for the Robotic Arm to become agentic. The arm was originally designed for this purpose, but we were unable to train with our limited time and computing power. There exist frameworks developed hugging face that will allow us to do this. Essentially, we would "train" the model by manually performing tasks using some kind of input, and the robot would eventually learn to emulate these on its own.


## Tech Stack

**Robotics & Hardware**
- LeRobot SO-101 Robotic Arm
- Servo-based 6-DOF control
- Wrist-mounted camera

**Backend**
- Python
- FastAPI
- Uvicorn
- Groq LLM API

**Computer Vision**
- YOLO
- OpenCV

**Frontend**
- Node.js
- React
- Web Speech API

**Other**
- 3D Printing
- Custom motion planning
