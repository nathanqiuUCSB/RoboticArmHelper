import os
from dotenv import load_dotenv
from lerobot.robots import make_robot

load_dotenv()

PORT = os.getenv("ROBOT_PORT")

robot = make_robot(robot_type="so_arm", port=PORT)
robot.connect()

obs = robot.get_observation()
print("Connected. Joint positions:", obs["joint_positions"])
