
#!/bin/bash

export CUDA_VISIBLE_DEVICES=3  # Use GPU 0
export PYTHONPATH="$PYTHONPATH:/root/code/open_lm"
export PYTHONPATH="$PYTHONPATH:/root/code/mixture_optimization"

cd /root/code/mixture_optimization
#torchrun --nproc_per_node 1 -m mixture_optimization.main --config_path config/config_deterministic_baseline.yaml
torchrun --nproc_per_node 1 -m mixture_optimization.main --experiment-dir logs/deterministic_baseline_human_intuition_v1_0