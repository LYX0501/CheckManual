import os

import imageio
import numpy as np
from PIL import Image
from sapien.core import Pose


cam_fix_mat = np.eye(4, dtype=np.float32)
cam_fix_mat[:3, :3] = np.array(
    [[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]],
    dtype=np.float32,
)


class EnvInterface:
    def __init__(self, env, robot, cam, task_dir):
        self.env = env
        self.robot = robot
        self.cam = cam
        self.save_dir = task_dir
        os.makedirs(self.save_dir, exist_ok=True)
        self.img_id = 0

    def get_observation(self):
        self.env.step()
        self.env.render()

        image, depth = self.cam.get_observation()
        meta = self.cam.get_metadata()
        cam_info = {
            "K": np.array(meta["camera_matrix"])[:3, :3],
            "H": image.shape[0],
            "W": image.shape[1],
            "scale": 1,
        }

        near, far = meta["near"], meta["far"]
        metric_depth = near * far / (far - (far - near) * depth).astype(np.float32)
        c2w = meta["mat44"] @ cam_fix_mat

        return {
            "image": (image * 255).astype(np.uint8),
            "depth": metric_depth,
            "c2w": c2w,
            "cam_info": cam_info,
            "normal": self.cam.get_normal_map(),
        }

    def move_pose(self, target_pose, type="set"):
        if type == "set":
            start_pose = Pose().from_transformation_matrix(target_pose)
            self.robot.robot.set_root_pose(start_pose)
            self.robot.wait_n_steps(2000)
            observation = self.get_observation()
            Image.fromarray(observation["image"]).save(
                os.path.join(self.save_dir, f"{self.img_id}_set_pose.png")
            )
        else:
            approach_imgs = self.robot.move_to_target_pose(
                target_pose,
                6000,
                cam=self.cam,
                vis_gif=True,
                vis_gif_interval=50,
                visu=False,
            )
            imageio.mimsave(
                os.path.join(self.save_dir, f"{self.img_id}_moving_pose.gif"),
                approach_imgs,
            )
            self.robot.wait_n_steps(2000)
            observation = self.get_observation()
            Image.fromarray(observation["image"]).save(
                os.path.join(self.save_dir, f"{self.img_id}_moved_pose.png")
            )
        self.img_id += 1

    def set_gripper_action(self, action):
        if action == "close":
            self.robot.close_gripper()
        else:
            self.robot.open_gripper()
        self.robot.wait_n_steps(1000)
        observation = self.get_observation()
        Image.fromarray(observation["image"]).save(
            os.path.join(self.save_dir, f"{self.img_id}_{action}_gripper.png")
        )
        self.img_id += 1

    def get_ee_pose(self, *args, **kwargs):
        return np.eye(4)

    def solve_ik(self, *args, **kwargs):
        return True, None
