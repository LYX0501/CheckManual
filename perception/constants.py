import os


DEFAULT_CHECKPOINT_DIR = os.environ.get(
    "CHECKMANUAL_CHECKPOINT_DIR",
    os.path.expanduser("~/.cache/checkmanual/checkpoints"),
)

DETIC_CHECKPOINT_PATH = os.environ.get(
    "CHECKMANUAL_DETIC_CHECKPOINT",
    os.path.join(DEFAULT_CHECKPOINT_DIR, "Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.pth"),
)
CLIP_CHECKPOINT_PATH = os.environ.get(
    "CHECKMANUAL_CLIP_CHECKPOINT",
    os.path.join(DEFAULT_CHECKPOINT_DIR, "clip-vit-base-patch16"),
)
GROUNDING_DINO_CONFIG_PATH = os.environ.get(
    "CHECKMANUAL_GROUNDING_DINO_CONFIG",
    os.path.join(DEFAULT_CHECKPOINT_DIR, "GroundingDINO_SwinT_OGC.py"),
)
GROUNDING_DINO_CHECKPOINT_PATH = os.environ.get(
    "CHECKMANUAL_GROUNDING_DINO_CHECKPOINT",
    os.path.join(DEFAULT_CHECKPOINT_DIR, "groundingdino_swint_ogc.pth"),
)
SAM_ENCODER_VERSION = "vit_h"
SAM_CHECKPOINT_PATH = os.environ.get(
    "CHECKMANUAL_SAM_CHECKPOINT",
    os.path.join(DEFAULT_CHECKPOINT_DIR, "sam_vit_h_4b8939.pth"),
)
RAM_MODEL = "swin_l"
RAM_CHECKPOINT_PATH = os.environ.get(
    "CHECKMANUAL_RAM_CHECKPOINT",
    os.path.join(DEFAULT_CHECKPOINT_DIR, "ram_swin_large_14m.pth"),
)
BLIP_MODEL = os.environ.get(
    "CHECKMANUAL_BLIP_MODEL",
    os.path.join(DEFAULT_CHECKPOINT_DIR, "instructblip-flanxl"),
)
LLAMA_CHECKPOINT_PATH = os.environ.get(
    "CHECKMANUAL_LLAMA_CHECKPOINT",
    os.path.join(DEFAULT_CHECKPOINT_DIR, "LLaMA-7B"),
)
CAPTION_CHECKPOINT_PATH = os.environ.get(
    "CHECKMANUAL_CAPTION_CHECKPOINT",
    os.path.join(DEFAULT_CHECKPOINT_DIR, "BIAS-7B.pth"),
)
INTEREST_ROOMS = [
    "livingroom",
    "kitchen",
    "diningroom",
    "hallway",
    "bedroom",
    "bathroom",
    "homeoffice",
    "laundryroom",
    "garage",
    "basement",
    "attic",
]
GLEE_CONFIG_PATH = os.environ.get(
    "CHECKMANUAL_GLEE_CONFIG",
    os.path.join(DEFAULT_CHECKPOINT_DIR, "GLEE", "configs", "SwinL.yaml"),
)
GLEE_CHECKPOINT_PATH = os.environ.get(
    "CHECKMANUAL_GLEE_CHECKPOINT",
    os.path.join(DEFAULT_CHECKPOINT_DIR, "GLEE", "GLEE_SwinL_Scaleup10m.pth"),
)
