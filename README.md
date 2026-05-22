# CheckManual: A New Challenge and Benchmark for Manual-based Appliance Manipulation

CheckManual is a benchmark for manual-based appliance manipulation. It provides
appliance manuals, task annotations, and simulator-based evaluation protocols for
three tracks:

- **Track 1:** manual-based part-function alignment and task planning.
- **Track 2:** manual-based primitive-action manipulation.
- **Track 3:** manual-based long-horizon manipulation with visual grounding and execution.

This repository contains the evaluation code and ManualPlan baselines. The
released datasets and model checkpoints are downloaded separately so the GitHub
repository stays lightweight.

<p align="center">
  <img src="docs/images/Teasor.jpg" style="width:80%;">
</p>

## News

- 2025.06.09: ManualPlan framework and Track 1 evaluation scripts released.
- 2025.06.05: [CheckManual dataset](https://drive.google.com/file/d/1YasM5Se7h4H8wCqZFN3mK8sCu1cEZBo7/view?usp=drive_link) released.
- 2025.04.04: CheckManual was selected as a CVPR 2025 Highlight.
- 2025.02.26: CheckManual was accepted by CVPR 2025.

## Repository Layout

```text
CheckManual/
├── track1_ManualPlan.py
├── track2_ManualPlan_fast.py      # recommended Track 2 entry point
├── track3_ManualPlan_fast.py      # recommended Track 3 entry point
├── track2_ManualPlan.py           # legacy/reference implementation
├── track3_ManualPlan.py           # legacy/reference implementation
├── api_utils/
├── manualplan_support/
├── perception/
├── voxposer/
├── robots/
├── assets/
├── data/                          # local datasets, ignored by git
└── results/                       # runtime outputs, ignored by git
```

The `fast` Track 2/3 scripts are the recommended public entry points. They keep
generated caches under `results/<run_name>/runtime_cache/` and are more suitable
for running from clean official data.

## Environment

The released scripts were tested with Python 3.8, CUDA-capable GPUs, and SAPIEN
0.8.0. A typical setup is:

```bash
conda create -n checkmanual python=3.8
conda activate checkmanual

pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 \
  torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117

sudo apt-get update
sudo apt-get install -y poppler-utils xvfb

pip install \
  http://download.cs.stanford.edu/orion/where2act/where2act_sapien_wheels/sapien-0.8.0.dev0-cp38-cp38-manylinux2014_x86_64.whl

pip install -r requirements.txt
```

Do not use the latest `pip install sapien` for these scripts. The code expects
the older SAPIEN 0.8.0 API used by Where2Act/PartNet-Mobility baselines.

Optional perception dependencies for Track 2 crop assistance and Track 3
predicted visual grounding:

```bash
pip install -r requirements-perception.txt
```

## Data Preparation

Download the [CheckManual manual dataset](https://drive.google.com/file/d/1YasM5Se7h4H8wCqZFN3mK8sCu1cEZBo7/view?usp=drive_link)
and the corresponding [PartNet-Mobility/SAPIEN appliance assets](https://sapien.ucsd.edu/downloads),
then arrange them as:

```text
data/
├── CheckManual_Data/
│   ├── manual_1/
│   │   ├── 100279_printer_manual_group1.pdf
│   │   ├── eval_tasks.json
│   │   └── part_state_functions.json
│   └── ...
└── sapien_dataset/
    ├── 100279/
    │   ├── mobility.urdf
    │   ├── semantics.txt
    │   ├── mobility_v2.json
    │   ├── meta.json
    │   └── textured_objs/
    └── ...
```

The public repository should not commit `data/CheckManual_Data/` or
`data/sapien_dataset/`. They are ignored by `.gitignore`.

The current public CheckManual release contains 1107 manual samples, 1484
manipulation tasks, 182 unique CAD shape ids, and 10 appliance categories:
camera, coffee_machine, dishwasher, display, microwave, oven, printer,
refrigerator, toaster, and washing_machine.

No pre-generated caches are required in `data/CheckManual_Data`. The scripts
generate PDF page images, local OCR fallbacks, visual alignment caches,
segmentation masks, and planning caches under:

```text
results/<run_name>/runtime_cache/<manual_xxx>/
```

## API Configuration

ManualPlan uses GPT/GPT-V for planning and multimodal alignment. Configure GPT
with environment variables:

```bash
export CHECKMANUAL_GPT_KEY="Bearer sk-..."
export CHECKMANUAL_GPT_URL="https://your-api-host/v1"
export CHECKMANUAL_GPT_MODEL="gpt-4o"
```

Alternatively, fill `api_utils/api_key_config.json`. Do not commit real API keys.

OCR is optional. If Baidu OCR is configured, set:

```bash
export CHECKMANUAL_OCR_API_KEY="..."
export CHECKMANUAL_OCR_SECRET_KEY="..."
```

If OCR is not configured, the scripts automatically fall back to local text
extraction with `pdftotext` and cache the result under `results/.../runtime_cache`.

## Quick Reproduction

These commands exercise the public code path without requiring GPT, OCR,
FoundationPose, or a running GroundingDINO/SAM server. They use oracle alignment
and oracle plans so you can verify the simulator, data layout, and result writer
first.

```bash
conda activate checkmanual

xvfb-run -a python track3_ManualPlan_fast.py \
  --manual_dir data/CheckManual_Data \
  --data_dir data/sapien_dataset \
  --sample manual_473 \
  --max_tasks 1 \
  --use_gt_alignment \
  --use_gt_plan \
  --out_dir results/smoke_track3_plan

xvfb-run -a python track3_ManualPlan_fast.py \
  --manual_dir data/CheckManual_Data \
  --data_dir data/sapien_dataset \
  --sample manual_473 \
  --max_tasks 1 \
  --use_gt_alignment \
  --use_gt_plan \
  --execute \
  --out_dir results/smoke_track3_exec
```

Expected output for `manual_473` is:

```text
Track 3 planning SR: 1.0000
Track 3 execution SR: 1.0000
```

After the oracle smoke tests pass, enable predicted GPT/GPT-V planning and the
perception services described below.

## Perception Services

### GroundingDINO + SAM

Predicted Track 3 visual grounding, Track 2 crop assistance, physical
button/knob interaction, and non-fallback Track 3 execution use the local
perception server:

```bash
export CHECKMANUAL_GROUNDING_DINO_CONFIG=/path/to/GroundingDINO_SwinT_OGC.py
export CHECKMANUAL_GROUNDING_DINO_CHECKPOINT=/path/to/groundingdino_swint_ogc.pth
export CHECKMANUAL_SAM_CHECKPOINT=/path/to/sam_vit_h_4b8939.pth
export CHECKMANUAL_CV_SERVER_PORT=5002

python perception/cv_server.py --port 5002
```

The defaults are defined in `perception/constants.py`, but environment variables
are recommended for portable setups.

If the server is not running, calls fail fast with a message such as
`CV server request failed: http://localhost:5002/sam`. Start the server or set
`CHECKMANUAL_CV_SERVER_PORT` to the port you are using.

### FoundationPose

Full Track 2 pose estimation requires FoundationPose. Install FoundationPose
following `perception/FoundationPose_Server/README.md`, then start:

```bash
python perception/FoundationPose_Server/foundationpose_flask.py
```

The Track 2 script sends requests to:

```text
http://127.0.0.1:6006/foundationpose_flask
```

Override it with `--foundationpose_url` if needed.

### AnyGrasp / Grasp Server

Track 2 slider, drawer, lid, and door execution can require grasp proposals. If
no cached grasp poses exist, provide a compatible grasp server with:

```bash
--grasp_server_url http://host:port/your_grasp_endpoint
```

Without a grasp server, those actions may be skipped or fail. Button and knob
actions do not need the grasp server.

## Track 1 Evaluation

Run a smoke test:

```bash
python track1_ManualPlan.py \
  --sample manual_473 \
  --max_tasks 1 \
  --out_dir results/smoke_track1
```

Run the full Track 1 evaluation:

```bash
python track1_ManualPlan.py \
  --manual_data_path data/CheckManual_Data \
  --data_dir data/sapien_dataset \
  --out_dir results/track1_full
```

Outputs:

```text
results/track1_full/track1_result.json
results/track1_full/runtime_cache/
```

## Track 2 Evaluation

`track2_ManualPlan_fast.py` is the recommended Track 2 entry point.

Smoke-test the simulator and result writing with oracle alignment and oracle
plans:

```bash
python track2_ManualPlan_fast.py \
  --manual_dir data/CheckManual_Data \
  --data_dir data/sapien_dataset \
  --sample manual_473 \
  --max_tasks 1 \
  --use_gt_alignment \
  --use_gt_plan \
  --out_dir results/smoke_track2_gt
```

Run Track 2 with predicted alignment and plans:

```bash
python track2_ManualPlan_fast.py \
  --manual_dir data/CheckManual_Data \
  --data_dir data/sapien_dataset \
  --foundationpose_url http://127.0.0.1:6006/foundationpose_flask \
  --grasp_server_url http://127.0.0.1:5000/grasp \
  --out_dir results/track2_full
```

Useful options:

- `--sample manual_473`: run one released sample.
- `--max_samples N`: run the first `N` samples.
- `--max_tasks N`: run the first `N` tasks per sample.
- `--no_cache_pose`, `--no_cache_plan`, `--no_cache_grasp`: force regeneration.
- `--use_gt_alignment`, `--use_gt_plan`: oracle smoke/debug modes.
- `--save_vis`: save per-step visualization images.

Outputs:

```text
results/track2_full/track2_results.json
results/track2_full/track2.log
results/track2_full/runtime_cache/
```

## Track 3 Evaluation

`track3_ManualPlan_fast.py` is the recommended Track 3 entry point.

Planning-only smoke test:

```bash
python track3_ManualPlan_fast.py \
  --manual_dir data/CheckManual_Data \
  --data_dir data/sapien_dataset \
  --sample manual_473 \
  --max_tasks 1 \
  --use_gt_alignment \
  --out_dir results/smoke_track3_plan
```

Track 3 execution requires GroundingDINO/SAM server first:

```bash
export CHECKMANUAL_CV_SERVER_PORT=5002
python perception/cv_server.py --port 5002
```

Then run:

```bash
python track3_ManualPlan_fast.py \
  --manual_dir data/CheckManual_Data \
  --data_dir data/sapien_dataset \
  --sample manual_473 \
  --max_tasks 1 \
  --execute \
  --out_dir results/smoke_track3_exec
```

The default Track 3 button/knob execution path uses joint-level fallbacks for
stability. It can run oracle button/knob smoke tests without SAM masks. Predicted
visual grounding, physical button/knob interaction, and door/slider-like actions
still require GroundingDINO/SAM masks.

Run full Track 3 planning:

```bash
python track3_ManualPlan_fast.py \
  --manual_dir data/CheckManual_Data \
  --data_dir data/sapien_dataset \
  --out_dir results/track3_full_plan
```

Run full Track 3 execution:

```bash
python track3_ManualPlan_fast.py \
  --manual_dir data/CheckManual_Data \
  --data_dir data/sapien_dataset \
  --execute \
  --out_dir results/track3_full_execute
```

For stability, the fast Track 3 script uses joint-level fallbacks for button and
knob execution by default. Physical button/knob interaction can be enabled with:

```bash
--try_physical_button_press
--try_physical_knob_rotate
```

Useful options:

- `--sample manual_473`: run one released sample.
- `--task_name "Task Name"`: run a single task by name.
- `--max_samples N`, `--max_tasks N`: limit evaluation size.
- `--use_gt_alignment`, `--use_gt_plan`: oracle debug modes.
- `--no_cache_alignment`, `--no_cache_plan`: force regeneration.
- `--save_vis`: save execution visualizations.
- `--no_sample_subprocess`: disable per-sample subprocess isolation. By default,
  `--execute` isolates each sample so native SAPIEN mesh-cooking failures are
  recorded as `sample_error` and do not stop the full batch.

Outputs:

```text
results/track3_full_execute/track3_results.json
results/track3_full_execute/track3.log
results/track3_full_execute/runtime_cache/
results/track3_full_execute/track3/
```

## Result Files

Each track writes a JSON result file under the chosen output directory.

Common fields:

- `total_tasks`: tasks evaluated for a sample.
- `success_task_plan`: tasks whose generated plan exactly matches ground truth.
- `success_task_execution`: tasks whose execution succeeds after matching the ground-truth plan.
- `completion_rates`: fraction of ground-truth steps completed during execution.
- `pred_link_function_dict`: predicted part/function alignment.
- `sample_error`: sample-level exception message, if a sample failed but the batch continued.

Runtime caches are stored separately under `runtime_cache/` so the released data
directory stays clean.

## Troubleshooting

- `ModuleNotFoundError` in `base`: activate `conda activate checkmanual` or use
  `conda run -n checkmanual ...`.
- `CV server request failed`: start `python perception/cv_server.py --port 5002`
  after configuring GroundingDINO/SAM checkpoint paths.
- Native SAPIEN segmentation faults: keep the default Track 3 sample subprocess
  isolation enabled. Failed samples are written with `sample_error` and the
  remaining batch continues.
- Empty OCR/API credentials: this is expected for oracle smoke tests. Full
  predicted ManualPlan runs need GPT/GPT-V credentials; OCR falls back to local
  `pdftotext` when Baidu OCR is not configured.

## Before Release Checklist

Before pushing a public repository, verify:

```bash
git status --short
```

Do not commit:

- `data/CheckManual_Data/`
- `data/sapien_dataset/`
- `results/`
- `__pycache__/` or `*.pyc`
- model checkpoints such as `*.pth`, `*.pt`, `*.ckpt`
- real API keys in `api_utils/api_key_config.json` or any log file

Recommended quick checks:

```bash
python -m py_compile track1_ManualPlan.py track2_ManualPlan_fast.py track3_ManualPlan_fast.py
rg -n "sk-[A-Za-z0-9_-]{20,}|OPENAI_API_KEY|CHECKMANUAL_GPT_KEY" .
```

## Citation

If you find CheckManual useful, please cite:

```bibtex
@inproceedings{checkmanual,
    author    = {Long, Yuxing and Zhang, Jiyao and Pan, Mingjie and Wu, Tianshu and Kim, Taewhan and Dong, Hao},
    title     = {CheckManual: A New Challenge and Benchmark for Manual-based Appliance Manipulation},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2025},
}
```
