# Implementation Planning: Edge of Post-Training Stability

Given the information amassed in `02_experiment_design`.md, there are a few key desiderata for the codebase:

Ideally, the codebase should leverage the fact that there is so much overlap between the three experiments; for example, we use the same general-purpose language model in all 3 experiments (tentatively GPT-2 Medium -> but should probably swap for something that is ONLY language model trained), use the same dataset for 2/3 categories (LoRA, full-param SFT), and can leverage much of that overlap to make implementation easier.

Some things I'd particularly care for:

1. Modular Implementation $\to$ since there is so much overlap in setup, I should be able to grab relevent dataloaders for each SFT/LoRA dataset (using the same ones as other staple papers in the niche) with a single function call. Should additionally be able to intialize the LM with a single function call shared across RLVR/SFT (need a different way to intialize the model for LoRA).

2. Flexibility $\to$ I plan to swap between notebook-style figure generation and command-line based plot generation for larger-scale experiments; starting with just the notebook approach for times' sake but should keep that in mind.

## File Structure and Setup

```
experiment_code/
├── notebooks/
│   ├── 00_sanity_checks.ipynb
│   ├── 01_sft_demo.ipynb
│   ├── 02_lora_demo.ipynb
│   └── 03_rlvr_demo.ipynb
│
├── scripts/
│   ├── run_sft.py
│   ├── run_lora.py
│   └── run_rlvr.py
│
├── src/
│   ├── dataloading.py
│   ├── arithmetic_gen.py
│   ├── models.py
│   └── utils.py.
│
├── requirements.txt
└── README.md
```

<!-- TODO: add implementation descriptions, file docstrings? etc. Doing this later though, seems like a 'cleanup for submission' task and not a 'getting the ball rolling' task.-->