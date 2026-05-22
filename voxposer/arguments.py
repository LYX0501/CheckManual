"""Load config for the lightweight SAPIEN VoxPoser integration."""
import os

try:
    import yaml
except ImportError:  # pragma: no cover - optional dependency
    yaml = None

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_VIS_DIR = os.path.join(ROOT_DIR, "results", "visualizations")

DEFAULT_RLBENCH_CONFIG = {
    "env_name": "rlbench",
    "planner": {
        "stop_threshold": 0.001,
        "savgol_polyorder": 3,
        "savgol_window_size": 21,
        "obstacle_map_weight": 1,
        "max_steps": 300,
        "obstacle_map_gaussian_sigma": 10,
        "target_map_weight": 2,
        "stop_criteria": "no_nearby_equal",
        "target_spacing": 1,
        "max_curvature": 3,
        "pushing_skip_per_k": 5,
    },
    "controller": {
        "horizon_length": 1,
        "num_samples": 10000,
        "ee_local": "temperature",
        "ee_local_radius": 0.15,
    },
    "visualizer": {
        "save_dir": DEFAULT_VIS_DIR,
        "quality": "low",
        "map_size": 100,
    },
    "lmp_config": {
        "env": {
            "map_size": 100,
            "num_waypoints_per_plan": 10000,
            "max_plan_iter": 1,
            "visualize": True,
        },
        "lmps": {
            "planner": {
                "prompt_fname": "planner_prompt",
                "model": "gpt-4o",
                "max_tokens": 512,
                "temperature": 0,
                "query_prefix": "# Query: ",
                "query_suffix": ".",
                "stop": ["# Query: ", "objects = "],
                "maintain_session": False,
                "include_context": True,
                "has_return": False,
                "return_val_name": "ret_val",
                "load_cache": True,
            },
            "composer": {
                "prompt_fname": "composer_prompt",
                "model": "gpt-4o",
                "max_tokens": 512,
                "temperature": 0,
                "query_prefix": "# Query: ",
                "query_suffix": ".",
                "stop": ["# Query: ", "objects ="],
                "maintain_session": False,
                "include_context": False,
                "has_return": False,
                "return_val_name": "ret_val",
                "load_cache": True,
            },
            "parse_query_obj": {
                "prompt_fname": "parse_query_obj_prompt",
                "model": "gpt-4o",
                "max_tokens": 512,
                "temperature": 0,
                "query_prefix": "# Query: ",
                "query_suffix": ".",
                "stop": ["# Query: ", "objects ="],
                "maintain_session": False,
                "include_context": True,
                "has_return": True,
                "return_val_name": "ret_val",
                "load_cache": True,
            },
            "get_affordance_map": {
                "prompt_fname": "get_affordance_map_prompt",
                "model": "gpt-4o",
                "max_tokens": 512,
                "temperature": 0,
                "query_prefix": "# Query: ",
                "query_suffix": ".",
                "stop": ["# Query: ", "objects ="],
                "maintain_session": False,
                "include_context": False,
                "has_return": True,
                "return_val_name": "ret_val",
                "load_cache": True,
            },
            "get_avoidance_map": {
                "prompt_fname": "get_avoidance_map_prompt",
                "model": "gpt-4o",
                "max_tokens": 512,
                "temperature": 0,
                "query_prefix": "# Query: ",
                "query_suffix": ".",
                "stop": ["# Query: ", "objects ="],
                "maintain_session": False,
                "include_context": False,
                "has_return": True,
                "return_val_name": "ret_val",
                "load_cache": True,
            },
            "get_velocity_map": {
                "prompt_fname": "get_velocity_map_prompt",
                "model": "gpt-4o",
                "max_tokens": 512,
                "temperature": 0,
                "query_prefix": "# Query: ",
                "query_suffix": ".",
                "stop": ["# Query: ", "objects ="],
                "maintain_session": False,
                "include_context": False,
                "has_return": True,
                "return_val_name": "ret_val",
                "load_cache": True,
            },
            "get_rotation_map": {
                "prompt_fname": "get_rotation_map_prompt",
                "model": "gpt-4o",
                "max_tokens": 512,
                "temperature": 0,
                "query_prefix": "# Query: ",
                "query_suffix": ".",
                "stop": ["# Query: ", "objects ="],
                "maintain_session": False,
                "include_context": False,
                "has_return": True,
                "return_val_name": "ret_val",
                "load_cache": True,
            },
            "get_gripper_map": {
                "prompt_fname": "get_gripper_map_prompt",
                "model": "gpt-4o",
                "max_tokens": 512,
                "temperature": 0,
                "query_prefix": "# Query: ",
                "query_suffix": ".",
                "stop": ["# Query: ", "objects ="],
                "maintain_session": False,
                "include_context": False,
                "has_return": True,
                "return_val_name": "ret_val",
                "load_cache": True,
            },
        },
    },
}


def load_config(config_path):
    with open(config_path, "r", encoding="utf-8") as file:
        if yaml is None:
            raise ImportError("PyYAML is not installed")
        return yaml.load(file, Loader=yaml.FullLoader)


def get_config(env=None, config_path=None):
    assert env is None or config_path is None, "env and config_path cannot be both specified"
    if config_path is None:
        assert env and env.lower() == "rlbench"
        config_path = os.path.join(ROOT_DIR, "voxposer", "configs", "rlbench_config.yaml")

    if config_path and os.path.exists(config_path) and yaml is not None:
        config = load_config(config_path)
    else:
        config = DEFAULT_RLBENCH_CONFIG

    class ConfigDict(dict):
        def __init__(self, config_dict):
            self.config = config_dict
            for key, value in config_dict.items():
                if isinstance(value, str) and value.lower() == "none":
                    value = None
                if isinstance(value, dict):
                    self[key] = ConfigDict(value)
                else:
                    self[key] = value

        def __getattr__(self, key):
            return self[key]

        def __setattr__(self, key, value):
            self[key] = value

        def __delattr__(self, key):
            del self[key]

        def __getstate__(self):
            return self.config

        def __setstate__(self, state):
            self.config = state
            self.__init__(state)

    return ConfigDict(config)
