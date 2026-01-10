import time
import os
from dotenv import load_dotenv
import numpy as np
from lerobot.robots import make_robot

load_dotenv()

PORT = os.getenv("ROBOT_PORT")


def smooth_send(robot, start, target, steps=40, dt=0.02):
    for a in np.linspace(0, 1, steps):
        action = (1 - a) * start + a * target
        robot.send_action(action)
        time.sleep(dt)


robot = make_robot(robot_type="so_arm", port=PORT)
robot.connect()

obs = robot.get_observation()
q0 = np.array(obs["joint_positions"], dtype=float)

q1 = q0.copy()
q1[0] += 0.10  # tiny move on joint 0

print("Moving joint 0 +0.10 rad...")
smooth_send(robot, q0, q1)

time.sleep(0.5)

print("Moving back...")
smooth_send(robot, q1, q0)

print("Done.")
