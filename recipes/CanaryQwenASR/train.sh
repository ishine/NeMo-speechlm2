#!/bin/bash
# SALM training script for Canary-Qwen-2.5B ASR model
# Auto-detects GPUs and nodes (-1 in config) or uses specified values
# Supports both single-node and multi-node training

ROOT_REPO="$(pwd)"
OUTDIR=$ROOT_REPO/recipes/CanaryQwenASR/outputs
YAML=salm_canary_qwen_2.5b

export PYTHONPATH=$ROOT_REPO

# MPI-based execution (auto-detects single vs multi-node)
# MPI will detect if running on single or multiple nodes automatically
mpirun --allow-run-as-root -bind-to none -map-by slot \
    -x no_proxy -x http_proxy -x https_proxy -x NCCL_DEBUG=INFO \
    -x PATH -x PYTHONPATH -x LD_LIBRARY_PATH -x S3_ENDPOINT_URL \
    -x S3_USE_HTTPS -x S3_ENDPOINT -x AWS_SECRET_ACCESS_KEY -x AWS_ACCESS_KEY_ID -x S3_VERIFY_SSL \
    -mca pml ob1 -mca btl ^openib --hostfile /horovod/generated/hostfile -mca orte_keep_fqdn_hostnames t \
/usr/bin/python3 $ROOT_REPO/recipes/CanaryQwenASR/speech_to_text_salm.py \
    --config-path=$ROOT_REPO/recipes/CanaryQwenASR/configs \
    --config-name=$YAML \
    exp_manager.exp_dir=$OUTDIR \
    "$@"  # Allow additional command-line overrides
