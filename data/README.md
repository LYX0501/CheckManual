# Data Layout

This directory intentionally keeps only documentation in the public repository.
Download the CheckManual dataset and PartNet-Mobility/SAPIEN assets separately,
then arrange them as follows:

```text
data/
├── CheckManual_Data/
│   ├── manual_1/
│   │   ├── 100279_printer_manual_group1.pdf
│   │   ├── eval_tasks.json
│   │   └── part_state_functions.json
│   ├── manual_2/
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

`CheckManual_Data` contains the released manual benchmark samples. Each sample
must include one PDF manual, `eval_tasks.json`, and `part_state_functions.json`.
The current public release contains 1107 manual samples and 1484 manipulation
tasks.

`sapien_dataset` contains the corresponding PartNet-Mobility/SAPIEN appliance
assets. At minimum each used shape id must include `mobility.urdf`,
`semantics.txt`, and all mesh files referenced by the URDF.
The current public release references 182 unique CAD shape ids across 10
categories: camera, coffee_machine, dishwasher, display, microwave, oven,
printer, refrigerator, toaster, and washing_machine.

Runtime caches are not required in the released data. The evaluation scripts
write generated files such as `manual_pngs/`, `link_pngs/`,
`all_masks_track3.pkl`, and `track*_plans.json` under
`results/<run_name>/runtime_cache/`.
