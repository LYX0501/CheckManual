import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from arguments import get_config
from interfaces import setup_LMP
from envs.real_env import VoxPoserReal
from vutils import set_lmp_objects

try:  # pragma: no cover - optional dependency
    from visualizers import ValueMapVisualizer
except ImportError:  # pragma: no cover - optional dependency
    ValueMapVisualizer = None




def run_voxposer(sapien_env, instruction, observation, category_ids=['door']):
    config = get_config('rlbench')
    visualizer = ValueMapVisualizer(config['visualizer']) if ValueMapVisualizer else None
    env = VoxPoserReal(visualizer=visualizer, obj_name_list=category_ids, init_obs=observation, sapien_env=sapien_env)
    lmps, lmp_env = setup_LMP(env, config, debug=False)
    voxposer_ui = lmps['plan_ui']
    set_lmp_objects(lmps, env.get_object_names())

    voxposer_ui(instruction)
