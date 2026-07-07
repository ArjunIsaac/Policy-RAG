#!/bin/bash
export CUDA_HOME=/usr
unset CUDA_PATH
export PATH="$VIRTUAL_ENV/bin:/usr/bin:$PATH"
export VLLM_USE_FLASHINFER_SAMPLER=0

echo "vLLM environment configured:"
echo "  python: $(which python3)"
echo "  VLLM_USE_FLASHINFER_SAMPLER=$VLLM_USE_FLASHINFER_SAMPLER"
