import os
import numpy as np
try:
    import open3d as o3d
except ImportError:
    import open3d_compat as o3d
import json
# from rlbench.action_modes.action_mode import MoveArmThenGripper
# from rlbench.action_modes.arm_action_modes import ArmActionMode, EndEffectorPoseViaPlanning
# from rlbench.action_modes.gripper_action_modes import Discrete, GripperActionMode
# from rlbench.environment import Environment
# import rlbench.tasks as tasks
# from pyrep.const import ObjectType
from vutils import normalize_vector, bcolors
import pickle
from scipy.spatial.transform import Rotation as R
import copy
import cv2
import time
# from omniagent.servers.detect.gdino_1_6 import run_detect, visual_detect_result


def _safe_artifact_name(name):
    return str(name).replace(os.sep, "_").replace("/", "_")

# 新env类的方法: 
# get_task:
    # task方法：
        # 提供ids（手动给就行）
        # reset
        # step时更新obs、reward、terminate

# TODO 机械臂控制方法
# TODO 拿到观测的方法

class Action():
    def __init__(self,sapien_env=None):
        # self.robot = Robot('/home/yan20/panmingjie/workspace/omniagent/omniagent/hardware/configs/franka_pku_umi.json') # 指定相机和机械臂的类型
        # self.robot.reset()
        # # self.robot.close_gripper()
        # print('reset robot')
        # self.last_gripper = 1

        self.sapien_env = sapien_env

    def action(self, action, action_type):   
        
        # action: xyz,rot,ee
        target_pose = np.eye(4)
        target_pose[:3,3] = action[:3]
        target_pose[:3,:3] = quat2mat(action[3:7])
        
        # self.robot.move_pose(target_pose,'Simple')
        # if self.last_gripper == action[7]:
        #     return
        # self.last_gripper = action[7]
        # if action[7] == 0:
        #     self.robot.close_gripper()
        #     print('close gripper')
        # else:
        #     self.robot.open_gripper()
        #     print('open gripper')
        
        if action_type == 'Set':
            try:
                print("set initial gripper!!!!!!!!!!!!!!!!")
                self.sapien_env.move_pose(target_pose,'set')
                print("complete setting initial gripper!!!!!!!!!!!!!!!!")
            except:
                return
        else:
            try:
                print('before action!!!!!!!!!!!!!!!!!!!!!!')
                self.sapien_env.move_pose(target_pose,'Simple')
                print('after action!!!!!!!!!!!!!!!!!!!!!!')
                if action[7] == 0:
                    self.sapien_env.set_gripper_action('close')
                    print('close gripper')
                else:
                    self.sapien_env.set_gripper_action('open')
                    print('open gripper')
            except:
                return



# from omniagent.hardware.franka.robot import Robot
# from omniagent.servers.pose.track_api import get_tracking_observation

def mat2quat(rotmat):
    # output 实部在前
    r = R.from_matrix(rotmat)
    quat_last = r.as_quat()
    quat_first = [quat_last[-1], quat_last[0], quat_last[1], quat_last[2]]
    return quat_first

def quat2mat(quat):
    # input 实部在前
    r = R.from_quat([quat[1], quat[2], quat[3], quat[0]])
    return r.as_matrix()

def depth_to_point_cloud(depth, camera_intrinsics):
    fx,fy,cx,cy = camera_intrinsics[0,0],camera_intrinsics[1,1],camera_intrinsics[0,2],camera_intrinsics[1,2]
    
    h, w = depth.shape # 创建网格 
    x_idx, y_idx = np.meshgrid(np.arange(w), np.arange(h)) 
    # 将像素坐标转换为相机坐标系中的点 
    Z = depth 
    X = (x_idx - cx) * Z / fx 
    Y = (y_idx - cy) * Z / fy 
    # 将X, Y, Z组合成一个h, w, 3的点云数组 
    point_cloud = np.stack((X, Y, Z), axis=-1) 
    return point_cloud


class RealEnv():
    def __init__(self,obj_name_list,init_obs,sapien_env):
        self.task = Task(obj_name_list=obj_name_list,init_obs=init_obs,sapien_env=sapien_env)
        init_obs = init_obs
        self.camera_c2w_dict = {'camera0': init_obs['c2w']}

    def get_task(self):
        return self.task
    
class Task():
    def __init__(self,obj_name_list,init_obs,sapien_env):
        self.obj_name_list = obj_name_list  # 一个list
        self._action_mode = Action(sapien_env)
        self.init_obs = init_obs
        self.sapien_env = sapien_env
        

    def step(self,action, action_type):

        self._action_mode.action(action, action_type)
        obs = self.get_obs()
        obs['gripper_pose'] = action[:7]

        reward = 0
        ternimate = 0
        return obs,reward,ternimate
    
    def get_obs(self):  # 写拿到状态的方法
        
        if self.init_obs is None:
            print('仅使用第一帧obs，code error！！！！！')
            
        
        # # while True:
        #     for i in range(3):
        #         obs = get_tracking_observation()
        #         time.sleep(3.0)
        #         depth = obs['depth']
        #         camera_intrinsics = obs['cam_info']['K']
        #         cam_space_points = depth_to_point_cloud(depth,camera_intrinsics)    # h,w,3
        #         world_space_points = (obs['c2w'][:3,:3] @ cam_space_points.reshape(-1,3).transpose(1,0)).transpose(1,0).reshape(depth.shape[0],depth.shape[1],3) + obs['c2w'][:3,3] 
        #         obs['points'] = world_space_points
                
        #         detect_info = run_detect(obs['image'],category=self.obj_name_list,segment=True,visual=True)
                
        #         detect_info['boxes'].append(detect_info['boxes'][0])
        #         detect_info['masks'][1] = detect_info['masks'][0]
        #         detect_info['categorys'].append('lid')
                
        #         obs['detect_info'] = detect_info
                
        #         cv2.imwrite('detect.png',visual_detect_result(detect_info))
                
        #         self.init_obs = obs
                
        #         if input('press enter to continue,print any word to detect again') == '':
        #             break

        else:
            obs = copy.deepcopy(self.init_obs)
            
            

        # gripper2base_mat = self._action_mode.robot.get_ee_pose()
        # quat = mat2quat(gripper2base_mat[:3,:3])
        # trans = gripper2base_mat[:3,3]
        # obs['gripper_pose'] = np.concatenate([trans,quat])
            

        
        
        # self.name2ids = {}
        # for i,name in enumerate(detect_info.keys()):
        #     self.name2ids[name] = i
        # print(self.name2ids)
        
        return obs
    


class VoxPoserReal():
    def __init__(self, visualizer=None,obj_name_list=None,init_obs=None,sapien_env=None):
        """
        Initializes the VoxPoserRLBench environment.
        改写时完全没考虑reset task

        Args:
            visualizer: Visualization interface, optional.
        """
        # action_mode = CustomMoveArmThenGripper(arm_action_mode=EndEffectorPoseViaPlanning(),
        #                                 gripper_action_mode=Discrete())
        # self.rlbench_env = Environment(action_mode)
        # self.rlbench_env.launch()

        # with open('/home/easter/wts/code/VoxPoser/observation.pkl','rb') as f:
        #     demo_obs = pickle.load(f)
        

        

        obj_name_list = obj_name_list

        self.real_env = RealEnv(obj_name_list,init_obs,sapien_env)
        init_obs = init_obs
        init_obs['gripper_open'] = 0.0
        self.init_obs = init_obs
        self.latest_obs = init_obs
        # self.task = None
        self.latest_action = None

        self.workspace_bounds_min = np.array([-5,-5,-5])
        self.workspace_bounds_max = np.array([5,5,5])
        self.visualizer = visualizer
        if self.visualizer is not None:
            self.visualizer.update_bounds(self.workspace_bounds_min, self.workspace_bounds_max)
            
        
        self.camera_names = ['camera0']     # 假设只有一个相机
        # calculate lookat vector for all cameras (for normal estimation)
        # name2cam = {
        #     'front': self.rlbench_env._scene._cam_front,
        # }
        forward_vector = np.array([0, 0, 1])
        self.lookat_vectors = {}
        for cam_name in self.camera_names:
            extrinsics = self.real_env.camera_c2w_dict[cam_name]
            lookat = extrinsics[:3, :3] @ forward_vector
            self.lookat_vectors[cam_name] = normalize_vector(lookat)
        # load file containing object names for each task
        # path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'task_object_names.json')
        # with open(path, 'r') as f:
        #     self.task_object_names = json.load(f)
        # self.task_object_names = self.real_env

        # self._reset_task_variables()
        self._update_visualizer()
        


    def get_object_names(self):
        """
        Returns the names of all objects in the current task environment.

        Returns:
            list: A list of object names.
        """
        # name_mapping = self.task_object_names[self.task.get_name()]
        # exposed_names = [names[0] for names in name_mapping]
        return self.real_env.task.obj_name_list

    # def load_task(self, task):
    #     """
    #     Loads a new task into the environment and resets task-related variables.
    #     Records the mask IDs of the robot, gripper, and objects in the scene.

    #     Args:
    #         task (str or rlbench.tasks.Task): Name of the task class or a task object.
    #     """
    #     # self._reset_task_variables()
    #     # if isinstance(task, str):
    #     #     task = getattr(tasks, task)
    #     self.task = self.real_env.get_task(task)
    #     # 这些mask的id应该都不用存，
    #     # self.arm_mask_ids = [obj.get_handle() for obj in self.task._robot.arm.get_objects_in_tree(exclude_base=False)]
    #     # self.gripper_mask_ids = [obj.get_handle() for obj in self.task._robot.gripper.get_objects_in_tree(exclude_base=False)]
    #     # self.robot_mask_ids = self.arm_mask_ids + self.gripper_mask_ids
    #     # self.obj_mask_ids = [obj.get_handle() for obj in self.task._task.get_base().get_objects_in_tree(exclude_base=False)]
    #     # store (object name <-> object id) mapping for relevant task objects
    #     # try:
    #     #     name_mapping = self.task_object_names[self.task.get_name()]
    #     # except KeyError:
    #     #     raise KeyError(f'Task {self.task.get_name()} not found in "envs/task_object_names.json" (hint: make sure the task and the corresponding object names are added to the file)')
    #     # exposed_names = [names[0] for names in name_mapping]
    #     # internal_names = [names[1] for names in name_mapping]
    #     # scene_objs = self.task._task.get_base().get_objects_in_tree(object_type=ObjectType.SHAPE,
    #     #                                                               exclude_base=False,
    #     #                                                               first_generation_only=False)
    #     # for scene_obj in scene_objs:
    #     #     if scene_obj.get_name() in internal_names:
    #     #         exposed_name = exposed_names[internal_names.index(scene_obj.get_name())]
    #     #         self.name2ids[exposed_name] = [scene_obj.get_handle()]
    #     #         self.id2name[scene_obj.get_handle()] = exposed_name
    #     #         for child in scene_obj.get_objects_in_tree():
    #     #             self.name2ids[exposed_name].append(child.get_handle())
    #     #             self.id2name[child.get_handle()] = exposed_name

    #     self.name2ids = {}
    #     for i in range(len(self.real_env.task.obj_name_list)):
    #         self.name2ids[self.real_env.task.obj_name_list[i]:i]

    def get_3d_obs_by_name(self, query_name):
        """
        Retrieves 3D point cloud observations and normals of an object by its name.

        Args:
            query_name (str): The name of the object to query.

        Returns:
            tuple: A tuple containing object points and object normals.
        """
        # # assert query_name in self.name2ids, f"Unknown object name: {query_name}"

        # if query_name not in self.real_env.task.obj_name_list:
        #     for detect_name in self.latest_obs['detect_info']['categorys']:
        #         if detect_name.replace('_',' ') in query_name:
        #             query_name = detect_name
        #             break
        #     print(f"{bcolors.WARNING}Object name not found")
        # query_name = query_name.replace(' ','_')
        # tmp = self.latest_obs['detect_info']['categorys']
        # assert query_name in self.latest_obs['detect_info']['categorys'],f'{tmp},query:{query_name}'
        # obj_id = self.latest_obs['detect_info']['categorys'].index(query_name)
        obj_id = 0
        
        # gather points and masks from all cameras
        obj_points_list, obj_normals_list = [], []
        for cam in self.camera_names:
            points = self.latest_obs['points'].reshape(-1, 3)
            masks = self.latest_obs['detect_info']['masks'][obj_id].reshape(-1).astype(bool)
            obj_points = points[masks]
            if len(obj_points) == 0:
                continue

            # Estimate normals only on the target-part point cloud instead of the
            # full scene cloud. This is both faster and more faithful to the queried
            # object observation.
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(obj_points)
            pcd.estimate_normals()
            cam_normals = np.asarray(pcd.normals)
            flip_indices = np.dot(cam_normals, self.lookat_vectors[cam]) > 0
            cam_normals[flip_indices] *= -1
            obj_points_list.append(obj_points)
            obj_normals_list.append(cam_normals)

        if not obj_points_list:
            raise ValueError(f"Object {query_name} not found in the scene")

        obj_points = np.concatenate(obj_points_list, axis=0)
        obj_normals = np.concatenate(obj_normals_list, axis=0)

        # voxel downsample using o3d
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(obj_points)
        o3d.io.write_point_cloud(
            os.path.join(
                self.real_env.task.sapien_env.save_dir,
                f"{_safe_artifact_name(query_name)}.ply",
            ),
            pcd,
        )
        
        pcd.normals = o3d.utility.Vector3dVector(obj_normals)
        pcd_downsampled = pcd.voxel_down_sample(voxel_size=0.001)
        
        
        obj_points = np.asarray(pcd_downsampled.points)
        obj_normals = np.asarray(pcd_downsampled.normals)
        return obj_points, obj_normals

    def get_scene_3d_obs(self, ignore_robot=False, ignore_grasped_obj=False):
        """
        Retrieves the entire scene's 3D point cloud observations and colors.

        Args:
            ignore_robot (bool): Whether to ignore points corresponding to the robot.
            ignore_grasped_obj (bool): Whether to ignore points corresponding to grasped objects.

        Returns:
            tuple: A tuple containing scene points and colors.
        """
        points, colors, masks = [], [], []
        for cam in self.camera_names:
            # points.append(self.latest_obs['points'].reshape(-1, 3))
            points.append(self.latest_obs['points'])
            colors.append(self.latest_obs['image'].reshape(-1, 3))
            # masks.append(self.latest_obs['mask'].reshape(-1))
        points = np.concatenate(points, axis=0)
        colors = np.concatenate(colors, axis=0)
        # masks = np.concatenate(masks, axis=0)

        # only keep points within workspace
        chosen_idx_x = (points[:, 0] > self.workspace_bounds_min[0]) & (points[:, 0] < self.workspace_bounds_max[0])
        chosen_idx_y = (points[:, 1] > self.workspace_bounds_min[1]) & (points[:, 1] < self.workspace_bounds_max[1])
        chosen_idx_z = (points[:, 2] > self.workspace_bounds_min[2]) & (points[:, 2] < self.workspace_bounds_max[2])
        points = points[(chosen_idx_x & chosen_idx_y & chosen_idx_z)]
        colors = colors[(chosen_idx_x & chosen_idx_y & chosen_idx_z)]
        # masks = masks[(chosen_idx_x & chosen_idx_y & chosen_idx_z)]

        # 直接不把gripper拍进去
        # if ignore_robot:
        #     robot_mask = np.isin(masks, self.robot_mask_ids)
        #     points = points[~robot_mask]
        #     colors = colors[~robot_mask]
        #     masks = masks[~robot_mask]
        # if self.grasped_obj_ids and ignore_grasped_obj:
        #     grasped_mask = np.isin(masks, self.grasped_obj_ids)
        #     points = points[~grasped_mask]
        #     colors = colors[~grasped_mask]
        #     masks = masks[~grasped_mask]

        # voxel downsample using o3d
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.colors = o3d.utility.Vector3dVector(colors)
        pcd_downsampled = pcd.voxel_down_sample(voxel_size=0.001)
        points = np.asarray(pcd_downsampled.points)
        colors = np.asarray(pcd_downsampled.colors).astype(np.uint8)

        return points, colors
        
        # return points, None

    # def reset(self):
    #     """
    #     Resets the environment and the task. Also updates the visualizer.

    #     Returns:
    #         tuple: A tuple containing task descriptions and initial observations.
    #     """
    #     assert self.task is not None, "Please load a task first"
    #     # self.task.sample_variation()
    #     descriptions, obs = self.task.reset()
    #     obs = self._process_obs(obs)
    #     self.init_obs = obs
    #     self.latest_obs = obs
    #     self._update_visualizer()
    #     return descriptions, obs




    def apply_action(self, action, action_type="Simple"):
        """
        Applies an action in the environment and updates the state.

        Args:
            action: The action to apply.  一个长度为8的array，四元数+gripper开闭

        Returns:
            tuple: A tuple containing the latest observations, reward, and termination flag.
        """
        # assert self.task is not None, "Please load a task first"
        action = self._process_action(action)
        obs, reward, terminate = self.real_env.task.step(action, action_type)
        obs = self._process_obs(obs)
        self.latest_obs = obs
        self.latest_reward = reward
        self.latest_terminate = terminate
        self.latest_action = action
        self._update_visualizer()
        # grasped_objects = self.rlbench_env._scene.robot.gripper.get_grasped_objects()
        # if len(grasped_objects) > 0:
            # self.grasped_obj_ids = [obj.get_handle() for obj in grasped_objects]
        return obs, reward, terminate

    def move_to_pose(self, pose, speed=None):
        """
        Moves the robot arm to a specific pose.

        Args:
            pose: The target pose.
            speed: The speed at which to move the arm. Currently not implemented.

        Returns:
            tuple: A tuple containing the latest observations, reward, and termination flag.
        """
        if self.latest_action is None:
            action = np.concatenate([pose, [self.init_obs['gripper_open']]])
        else:
            action = np.concatenate([pose, [self.latest_action[-1]]])
        return self.apply_action(action)
    
    def open_gripper(self):
        """
        Opens the gripper of the robot.
        """
        action = np.concatenate([self.latest_obs['gripper_pose'], [1.0]])
        return self.apply_action(action)

    def close_gripper(self):
        """
        Closes the gripper of the robot.
        """
        action = np.concatenate([self.latest_obs['gripper_pose'], [0.0]])
        return self.apply_action(action)

    def set_gripper_state(self, gripper_state):
        """
        Sets the state of the gripper.

        Args:
            gripper_state: The target state for the gripper.

        Returns:
            tuple: A tuple containing the latest observations, reward, and termination flag.
        """
        action = np.concatenate([self.latest_obs['gripper_pose'], [gripper_state]])
        return self.apply_action(action)

    def reset_to_default_pose(self):
        """
        Resets the robot arm to its default pose.

        Returns:
            tuple: A tuple containing the latest observations, reward, and termination flag.
        """
        if self.latest_action is None:
            action = np.concatenate([self.init_obs['gripper_pose'], [self.init_obs['gripper_open']]])
        else:
            action = np.concatenate([self.init_obs['gripper_pose'], [self.latest_action[-1]]])
        return self.apply_action(action, action_type="Set")

    def get_ee_pose(self):
        assert self.latest_obs is not None, "Please reset the environment first"
        return self.latest_obs['gripper_pose']

    def get_ee_pos(self):
        return self.get_ee_pose()[:3]

    def get_ee_quat(self):
        return self.get_ee_pose()[3:]

    def get_last_gripper_action(self):
        """
        Returns the last gripper action.

        Returns:
            float: The last gripper action.
        """
        if self.latest_action is not None:
            return self.latest_action[-1]
        else:
            return self.init_obs['gripper_open']

    # def _reset_task_variables(self):
    #     """
    #     Resets variables related to the current task in the environment.

    #     Note: This function is generally called internally.
    #     """
    #     self.init_obs = None
    #     self.latest_obs = None
    #     self.latest_reward = None
    #     self.latest_terminate = None
    #     self.latest_action = None
    #     self.grasped_obj_ids = None
    #     # scene-specific helper variables
    #     self.arm_mask_ids = None
    #     self.gripper_mask_ids = None
    #     self.robot_mask_ids = None
    #     self.obj_mask_ids = None
    #     self.name2ids = {}  # first_generation name -> list of ids of the tree
    #     self.id2name = {}  # any node id -> first_generation name
   
    def _update_visualizer(self):
        """
        Updates the scene in the visualizer with the latest observations.

        Note: This function is generally called internally.
        """
        if self.visualizer is not None:
            points, colors = self.get_scene_3d_obs(ignore_robot=False, ignore_grasped_obj=False)
            self.visualizer.update_scene_points(points, colors)
    
    def _process_obs(self, obs):
        """
        Processes the observations, specifically converts quaternion format from xyzw to wxyz.
        这两个函数发什么神经？
        Args:
            obs: The observation to process.

        Returns:
            The processed observation.
        """
        # quat_xyzw = obs.gripper_pose[3:]
        # quat_wxyz = np.concatenate([quat_xyzw[-1:], quat_xyzw[:-1]])
        # obs.gripper_pose[3:] = quat_wxyz
        return obs

    def _process_action(self, action):
        """
        Processes the action, specifically converts quaternion format from wxyz to xyzw.

        Args:
            action: The action to process.

        Returns:
            The processed action.
        """
        # quat_wxyz = action[3:7]
        # quat_xyzw = np.concatenate([quat_wxyz[1:], quat_wxyz[:1]])
        # action[3:7] = quat_xyzw
        return action
