import sys
import os
code_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(f'{code_dir}/FoundationPose')
os.environ['PYOPENGL_PLATFORM'] = 'egl'
from Utils import *
import json,uuid,joblib,os,sys,argparse
from datareader import *
from estimater import *

import yaml
import time
from torch.utils.data import DataLoader

from datetime import datetime
from flask import Flask, request, jsonify
import pyrender
import pickle

import pybullet as p
import pybullet_data
torch.set_default_dtype(torch.float32)

class Renderer:
    def __init__(self, width, height, camera_intrinsic):
        self.width = width
        self.height = height
        fx, fy = camera_intrinsic['fx'], camera_intrinsic['fy']
        cx, cy = camera_intrinsic['cx'], camera_intrinsic['cy']
        
        self.renderer = pyrender.OffscreenRenderer(width, height)
        self.camera = pyrender.IntrinsicsCamera(fx, fy, cx, cy)
        
        self.pyrender_scene = pyrender.Scene()
        
        camera_pose = np.eye(4)
        camera_pose[1, 1] = -1
        camera_pose[2, 2] = -1
        self.pyrender_scene.add(self.camera, pose=camera_pose)
        light = pyrender.DirectionalLight(color=np.ones(3) * 0.8, intensity=5.0)
        self.pyrender_scene.add(light, pose=camera_pose)
    
    def vis_pose_pyrender(self, rgb, pred_pose, mesh):
        image = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        height, width, _ = image.shape
        
        mesh_pyrender = pyrender.Mesh.from_trimesh(mesh)
        mesh_node = self.pyrender_scene.add(mesh_pyrender, pose=pred_pose)

        color, depth = self.renderer.render(self.pyrender_scene, flags=pyrender.RenderFlags.SKIP_CULL_FACES)
        
        self.pyrender_scene.remove_node(mesh_node)
        
        mask = depth > 0
        result = image.copy()
        result[mask] = image[mask] * 0.3 + [75, 75, 0]
        
        mask = np.array(mask, dtype=np.uint8)
        kernel = np.ones((3, 3), np.uint8)
        dilated_mask = cv2.dilate(mask, kernel, iterations=2)
        edge_mask = dilated_mask - mask
        result[edge_mask > 0] = [255, 255, 255]

        return result.astype(np.uint8),color.astype(np.uint8)

app = Flask(__name__)
def get_mask(reader, i_frame, ob_id, detect_type):
  if detect_type=='box':
    mask = reader.get_mask(i_frame, ob_id)
    H,W = mask.shape[:2]
    vs,us = np.where(mask>0)
    umin = us.min()
    umax = us.max()
    vmin = vs.min()
    vmax = vs.max()
    valid = np.zeros((H,W), dtype=bool)
    valid[vmin:vmax,umin:umax] = 1
  elif detect_type=='mask':
    mask = reader.get_mask(i_frame, ob_id, type='mask_visib')
    valid = mask>0
  elif detect_type=='cnos':   #https://github.com/nv-nguyen/cnos
    mask = cv2.imread(reader.color_files[i_frame].replace('rgb','mask_cnos'), -1)
    valid = mask==ob_id
  else:
    raise RuntimeError

  return valid



def run_pose_estimation_worker(reader, i_frames, est:FoundationPose, debug=False, ob_id=None, device:int=0):
  result = NestDict()
  debug_dir = est.debug_dir

  for i in range(len(i_frames)):
    i_frame = i_frames[i]
    id_str = reader.id_strs[i_frame]
    logging.info(f"{i}/{len(i_frames)}, video:{reader.get_video_id()}, id_str:{id_str}")
    color = reader.get_color(i_frame)
    depth = reader.get_depth(i_frame)

    H,W = color.shape[:2]
    scene_ob_ids = reader.get_instance_ids_in_image(i_frame)
    video_id = reader.get_video_id()

    logging.info(f"video:{reader.get_video_id()}, id_str:{id_str}, ob_id:{ob_id}")
    if ob_id not in scene_ob_ids:
      logging.info(f'skip {ob_id} as it does not exist in this scene')
      continue
    ob_mask = get_mask(reader, i_frame, ob_id, detect_type=detect_type)

    est.gt_pose = reader.get_gt_pose(i_frame, ob_id)
    logging.info(f"pose:\n{pose}")
    exit()

    if debug>=3:
      tmp = est.mesh_ori.copy()
      tmp.apply_transform(pose)
      tmp.export(f'{debug_dir}/model_tf.obj')

    result[video_id][id_str][ob_id] = pose

  return result

def depth_map_to_pointcloud(depth_map, mask, intrinsics):
    # Get dimensions
    H, W = depth_map.shape
    
    if mask is not None:
        depth_map[mask == 0] = -1
    
    if torch.is_tensor(depth_map):
        depth_map = depth_map.cpu().numpy()
    
    # Unpack intrinsic matrix
    fx = torch.tensor(intrinsics['fx']).item()
    fy = torch.tensor(intrinsics['fy']).item()
    cx = torch.tensor(intrinsics['cx']).item()
    cy = torch.tensor(intrinsics['cy']).item()
    
    # Create grid of pixel coordinates
    u, v = np.meshgrid(np.arange(W), np.arange(H))
    
    # Convert pixel coordinates to camera coordinates
    x = (u - cx) * depth_map / fx
    y = (v - cy) * depth_map / fy
    z = depth_map
    
    # Reshape to (B*S, H*W)
    x = np.reshape(x, (-1))
    y = np.reshape(y, (-1))
    z = np.reshape(z, (-1))
    
    # Stack into point cloud
    pointcloud = np.stack((x, y, z), axis=-1)
    
    pointcloud = pointcloud[pointcloud[:,2] > 0]
    
    return pointcloud
  
def transform_pointcloud(pointcloud, transformation_matrix):
    
    # Append a column of ones to make homogeneous coordinates
    homogeneous_points = np.hstack((pointcloud, np.ones((pointcloud.shape[0], 1))))
    
    # Perform transformation
    transformed_points = np.dot(transformation_matrix, homogeneous_points.T).T
    
    # Divide by the last coordinate (homogeneous division)
    transformed_points = transformed_points[:, :3] / transformed_points[:, 3][:, np.newaxis]
    
    return transformed_points
  
def save_pointcloud(pointcloud, filename):
    """
    Save a point cloud to a text file.

    Args:
        pointcloud (numpy array): Point cloud array of shape (N, 3).
        filename (str): Name of the file to save.
    """
    if torch.is_tensor(pointcloud):
        pointcloud = pointcloud.cpu().numpy()
    if pointcloud.shape[1] == 3:
        with open(filename, 'w') as f:
            for point in pointcloud:
                f.write(f"{point[0]} {point[1]} {point[2]}\n")
    if pointcloud.shape[1] == 6:
        with open(filename, 'w') as f:
            for point in pointcloud:
                f.write(f"{point[0]} {point[1]} {point[2]} {point[3]} {point[4]} {point[5]}\n")

     
def read_pointcloud(filename):
    f = open(filename)
    lines = f.readlines()
    pointcloud = []
    for line in lines:
        x, y, z = float(line.split(' ')[0]), float(line.split(' ')[1]), float(line.split(' ')[2])
        pointcloud.append([x, y, z])
    pointcloud = np.array(pointcloud)
    f.close()
    return pointcloud             

class PoseRunner():

  def __init__(self,):

    wp.force_load(device='cuda')
    debug = opt.debug
    use_reconstructed_mesh = opt.use_reconstructed_mesh
    debug_dir = opt.debug_dir
    
    glctx = dr.RasterizeCudaContext()
    mesh_tmp = trimesh.primitives.Box(extents=np.ones((3)), transform=np.eye(4))
    self.est = FoundationPose(model_pts=mesh_tmp.vertices.copy(), model_normals=mesh_tmp.vertex_normals.copy(), symmetry_tfs=None, mesh=mesh_tmp, scorer=None, refiner=None, glctx=glctx, debug_dir=debug_dir, debug=debug)
    
    self.renderer = None

  def reset(self,mesh_path):

      self.mesh = self.load_model(mesh_path) 
      
      symmetry_tfs = [
        [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]],
        [[-1,0,0,0],[0,-1,0,0],[0,0,1,0],[0,0,0,1]]
      ]
      self.est.reset_object(model_pts=self.mesh.vertices.copy(), model_normals=self.mesh.vertex_normals.copy(), symmetry_tfs=symmetry_tfs, mesh=self.mesh)
      self.est.diameter = self.est.diameter.astype('float32')
      device = opt.device
      torch.cuda.set_device(device)
      self.est.to_device(f'cuda:{device}')
      self.est.glctx = dr.RasterizeCudaContext(device)
      
  def load_model(self,path):
    meshes = []
    for filename in os.listdir(path):
        if filename.endswith('.obj'):
            mesh_path = os.path.join(path, filename)
            mesh = trimesh.load(mesh_path)
            meshes.append(mesh)

    combined_mesh = trimesh.util.concatenate(meshes)

    return combined_mesh

  

  def run(self,rgb=None,depth=None,mask=None,intrinsic=None,mesh_path=None,first_flag=True):
      
    self.reset(mesh_path)

    self.K = intrinsic
    ob_id = 1
    if first_flag:
      pose = self.est.register(K=self.K, rgb=rgb, depth=depth, ob_mask=mask, ob_id=ob_id, iteration=5)
      print('first frame')
    else:
      time0 = time.time()
      pose = self.est.track_one(rgb=rgb, depth=depth, K=self.K, iteration=2)
      time1 = time.time()
      print('tracking time',time1-time0)
    return pose
  
  def get_joint2obj(self,urdf_path,worldurdf2obj):
        
        
        p.connect(p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        obj_urdf = p.loadURDF(urdf_path)
        
        numJoints = p.getNumJoints(obj_urdf)
        linkname_jointidx_dict = {}
        joint2obj_dict = {}
        for joint_idx in range(numJoints):
            joint_info = p.getJointInfo(obj_urdf, joint_idx)
            joint_axis_local = joint_info[13]  # joint2part
            if joint_axis_local == (0,0,0):
                continue
              
            joint_child_link = joint_info[12].decode('utf-8')
            linkname_jointidx_dict[joint_child_link] = joint_idx
            
            initial_link_state = p.getLinkState(obj_urdf, joint_idx) 
              
            joint_world_position = initial_link_state[0]
            joint_world_orientation = initial_link_state[1]   # part2world
            
            world_rotation_matrix = p.getMatrixFromQuaternion(joint_world_orientation)
            joint_axis_world = np.dot(np.array(world_rotation_matrix).reshape(3, 3), joint_axis_local)  # part2world @ joint2part = joint2world
            joint_axis_world = joint_axis_world / np.linalg.norm(joint_axis_world)
            # worldurdf2obj @ joint2worldurdf = joint2obj 
            joint2obj_dict[joint_child_link] = {
              'axis2obj':worldurdf2obj[:3,:3]@joint_axis_world,   
              't2obj':worldurdf2obj[:3,:3]@joint_world_position+worldurdf2obj[:3,3]  
            }
            
        return joint2obj_dict
            

      
  def vis_joint_axis(self, rgb, joint2cam_dict, camera_intrinsic):
      camera_matrix = np.array([
          [camera_intrinsic['fx'], 0, camera_intrinsic['cx']],
          [0, camera_intrinsic['fy'], camera_intrinsic['cy']],
          [0, 0, 1]
      ])
      
      for joint_name, joint_data in joint2cam_dict.items():
          axis2cam = np.array(joint_data['axis2cam']) 
          t2cam = np.array(joint_data['t2cam'])  
          
          axis_length = 1.0

          axis_points_3d = np.float32([
              t2cam,             
              t2cam + axis2cam * axis_length
          ])
          rotation_vector = np.zeros((3, 1))
          translation_vector = np.zeros((3, 1))
          
          axis_points_2d, _ = cv2.projectPoints(
              axis_points_3d,
              rotation_vector,
              translation_vector,
              camera_matrix,
              np.zeros(5)
          )

          axis_points_2d = np.int32(axis_points_2d).reshape(-1, 2)
          origin = tuple(axis_points_2d[0])
          end = tuple(axis_points_2d[1]) 

          cv2.line(rgb, origin, end, (0, 0, 255), 2)

      return rgb
      
@app.route('/foundationpose_flask', methods=['POST'])
def foundationpose_flask():
  print('getting data')
  data = request.get_json()
  first_flag = True
  cam_metadata = data.get('cam_metadata')
  rgbdmask = data.get('rgbdmask')
  path = data.get('path')
  print('done!')
  
  camera_matrix = np.array(cam_metadata['camera_matrix'])[:3,:3].astype(np.float32)
  camera_intrinsic = {'fx':camera_matrix[0,0],'cx':camera_matrix[0,2],'fy':camera_matrix[1,1],'cy':camera_matrix[1,2]}
  near = cam_metadata['near']
  far = cam_metadata['far']
  
  rgb = (np.array(rgbdmask['rgb'])*255).astype(np.uint8)
  depth_rela = np.array(rgbdmask['depth'])
  depth = near*far/(far+depth_rela*(near-far)).astype(np.float32) 
  mask = np.array(rgbdmask['obj_mask'])
  
  obj2cam = runner.run(rgb,depth,mask,camera_matrix,mesh_path=f'{path}/textured_objs')

  # cam_metadata['model_matrix'] is obj2urdf
  joint2obj_dict = runner.get_joint2obj(f'{path}/mobility.urdf',worldurdf2obj=np.linalg.inv(cam_metadata['model_matrix']))
  
  joint2cam_dict = {}
  for joint_name in joint2obj_dict.keys():
      axis2obj = joint2obj_dict[joint_name]['axis2obj']
      t2obj = joint2obj_dict[joint_name]['t2obj']
      joint2cam_dict[joint_name] = {
              'axis2cam':(obj2cam[:3,:3] @ axis2obj).tolist(),
              't2cam':(obj2cam[:3,:3] @ t2obj + obj2cam[:3,3]).tolist()
            }
  
  camset2world = np.array(cam_metadata['mat44'])   # x:forward, z:up
  # need to change to z:forward，-y:up
  camrender2camset = np.eye(4)
  camrender2camset[:3,:3] = np.array([[0,0,1],
                                      [-1,0,0],
                                      [0,-1,0]])
  cam2world = camset2world @ camrender2camset
  
  joint2world_dict = {}
  for joint_name in joint2cam_dict.keys():
          axis2cam = np.array(joint2cam_dict[joint_name]['axis2cam'])
          t2cam = np.array(joint2cam_dict[joint_name]['t2cam'])
          joint2world_dict[joint_name] = {
                  'axis2world':(cam2world[:3,:3] @ axis2cam).tolist(),              
                  't2world':(cam2world[:3,:3] @ t2cam + cam2world[:3,3]).tolist() 
                }

  print('rgb:',rgb.shape)
  print('depth:',depth.shape)
  print('mask:',mask.shape)
  if runner.renderer is None:
        runner.renderer = Renderer(rgb.shape[1],rgb.shape[0],camera_intrinsic)
  render_result,render_color = runner.renderer.vis_pose_pyrender(rgb,obj2cam,runner.mesh)
  
  axis_result = runner.vis_joint_axis(render_result,joint2cam_dict,camera_intrinsic)
  result_name = str(path).split('/')[-1]
  print('save visiualization result to: ',f'/root/autodl-tmp/wts/tmp_vis/{result_name}.png')
  cv2.imwrite(f'/root/autodl-tmp/wts/tmp_vis/{result_name}.png',axis_result) 
  cv2.imwrite(f'/root/autodl-tmp/wts/tmp_vis/{result_name}_color.png',render_color) 
  
  return_dict = {
    'joint2cam_dict':joint2cam_dict,
    'joint2world_dict':joint2world_dict,
    'obj2world': (cam2world @ obj2cam).tolist()
  }
  
  print('obj2world:',cam2world @ obj2cam)
  
  return jsonify(return_dict)




if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  code_dir = os.path.dirname(os.path.realpath(__file__))
  #   parser.add_argument('--sim_dir', type=str, default="/root/autodl-tmp/shiqian/code/gripper/test_views/franka_69.4_64", help="data dir")
  parser.add_argument('--use_reconstructed_mesh', type=int, default=0)
  # parser.add_argument('--ref_view_dir', type=str, default="/mnt/9a72c439-d0a7-45e8-8d20-d7a235d02763/DATASET/YCB_Video/bowen_addon/ref_views_16")
  parser.add_argument('--debug', type=int, default=0)
  parser.add_argument('--debug_dir', type=str, default=f'{code_dir}/debug_sim_vids')
  # parser.add_argument('--gripper', type=str, default="heph")
  parser.add_argument('--device', type=int, default=0)
  opt = parser.parse_args()
  #   os.environ["SIM_DATA_DIR"] = opt.sim_dir

  set_seed(0)

  detect_type = 'mask'   # mask / box / detected
  runner = PoseRunner()

  # AutoDL only opens '6006' port for external request.
  app.run(host='0.0.0.0', port=6006)

