"""RoboTwin policy that ends every episode at once. Run through script/eval_policy.py it reproduces the evaluator's
scene stream (seeds 100000*(1+k) upward, expert-checked) without a model; it saves each episode's first head frame
so the code arm's scenes can be checked against it."""
import os

import numpy as np
from PIL import Image


def get_model(usr_args):
    return {"out": os.environ["SEEDSCAN_FRAMES"], "n": 0}


def reset_model(model):
    model["first"] = True


def eval(TASK_ENV, model, observation):
    if model["first"]:
        rgb = np.asarray(observation["observation"]["head_camera"]["rgb"], dtype=np.uint8)   # [H, W, 3]
        Image.fromarray(rgb).save(f"{model['out']}/ep{model['n']:03d}.png")
        model["n"] += 1
        model["first"] = False
    TASK_ENV.take_action_cnt = TASK_ENV.step_lim
