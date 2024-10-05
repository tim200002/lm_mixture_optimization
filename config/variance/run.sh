#!/bin/bash

export CUDA_VISIBLE_DEVICES=2  # Use GPU 0
export PYTHONPATH="$PYTHONPATH:/root/code/open_lm"
export PYTHONPATH="$PYTHONPATH:/root/code/mixture_optimization"

cd /root/code/mixture_optimization
python -m mixture_optimization.main --config-path config/variance/config_var2_simplex_cc_books_stack_pes2o_reddit.yaml
python -m mixture_optimization.main --config-path config/variance/config_var3_simplex_cc_books_stack_pes2o_reddit.yaml

python -m mixture_optimization.main --config-path config/variance/config_var4_dolma_weights_cc_books_stack_pes2o_reddit.yaml
python -m mixture_optimization.main --config-path config/variance/config_var5_dolma_weights_cc_books_stack_pes2o_reddit.yaml
