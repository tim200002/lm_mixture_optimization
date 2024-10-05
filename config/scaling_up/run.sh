#!/bin/bash

export CUDA_VISIBLE_DEVICES= # Use GPU 0
export PYTHONPATH="$PYTHONPATH:/root/code/open_lm"
export PYTHONPATH="$PYTHONPATH:/root/code/mixture_optimization"

cd /root/code/mixture_optimization

#python -m mixture_optimization.main --experiment-dir logs/uniform_scale_up_154m_cc_books_stack_pes2o_reddit_0
python -m mixture_optimization.main --experiment-dir logs/uniform_scale_up_154m_cc_books_stack_pes2o_reddit_1
