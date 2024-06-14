
#!/bin/bash

export CUDA_VISIBLE_DEVICES=3  # Use GPU 0
export PYTHONPATH="$PYTHONPATH:/root/code/open_lm"
torchrun --nproc_per_node 1 -m open_lm.main --train-num-samples 16000000 --workers 1 --dataset-manifest /media/ssd1/tim/data_workspace/Test_12/run_0/manifest.jsonl --global-batch-size 10 --log-every-n-steps 50 --grad-clip-norm 1 --lr 0.0003 --warmup 200 --wd 0.1 --beta2 0.95 --epochs 5 --report-to tensorboard --data-key txt --lr-cooldown-end 3e-05 --logs-dir /root/code/mixture_optimization/logs/Test_33/runs/run_0/open_lm --name run_