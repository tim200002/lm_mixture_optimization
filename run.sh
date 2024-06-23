
#!/bin/bash

export CUDA_VISIBLE_DEVICES=  # Use GPU 0
export PYTHONPATH="$PYTHONPATH:/root/code/open_lm"
export PYTHONPATH="$PYTHONPATH:/root/code/mixture_optimization"

cd /root/code/mixture_optimization
#python -m mixture_optimization.main --config_path config/config_bayesian.yaml
python -m mixture_optimization.main --experiment-dir logs/deterministic_baseline_human_intuition_v1_4
#python -m mixture_optimization.main --config-path config/config_deterministic_baseline.yaml