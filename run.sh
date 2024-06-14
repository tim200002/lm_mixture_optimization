
#!/bin/bash

export CUDA_VISIBLE_DEVICES=3  # Use GPU 0
export PYTHONPATH="$PYTHONPATH:/root/code/open_lm"
export PYTHONPATH="$PYTHONPATH:/root/code/mixture_optimization"

cd /root/code/mixture_optimization
torchrun --nproc_per_node 1 -m mixture_optimization.main 