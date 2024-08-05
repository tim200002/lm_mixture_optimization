#!/bin/bash

export CUDA_VISIBLE_DEVICES=2  # Use GPU 0
export PYTHONPATH="$PYTHONPATH:/root/code/open_lm"
export PYTHONPATH="$PYTHONPATH:/root/code/mixture_optimization"

cd /root/code/mixture_optimization
#python -m mixture_optimization.main --config_path config/config_bayesian.yaml
#python -m mixture_optimization.main --experiment-dir logs/turbo-new_15
#python -m mixture_optimization.main --experiment-dir logs/bayesian_new_1
#python -m mixture_optimization.main --experiment-dir logs/turbo-new-bounds_1
#python -m mixture_optimization.main --config-path config/config_turbo.yaml
#python -m mixture_optimization.main --config-path config/config_lin_interp_2_sources.yaml
#python -m mixture_optimization.main --experiment-dir logs/lin_interp_books_c4_0
#python -m mixture_optimization.main --config-path config/config_turbo_two_sources_79m.yaml
#python -m mixture_optimization.main --config-path config/config_turbo_two_sources_154m.yaml
#python -m mixture_optimization.main --experiment-dir logs/bayesian_books_cc_stack_0
python -m mixture_optimization.main --experiment-dir logs/bayesian_books_cc_stack_pes2o_16init_0
#python -m mixture_optimization.main --config-path config/config_simplex_cc_books_stack_pes2o.yaml