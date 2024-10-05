#!/bin/bash

export CUDA_VISIBLE_DEVICES=  # Use GPU 0
export PYTHONPATH="$PYTHONPATH:/root/code/open_lm"
export PYTHONPATH="$PYTHONPATH:/root/code/mixture_optimization"

cd /root/code/mixture_optimization
python -m mixture_optimization.main --experiment-dir logs/variance_test_simplex_2_cc_books_stack_pes2o_reddit_0
python -m mixture_optimization.main --experiment-dir logs/variance_test_simplex_3_cc_books_stack_pes2o_reddit_0