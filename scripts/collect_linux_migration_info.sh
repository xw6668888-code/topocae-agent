#!/usr/bin/env bash

set -u

target_path="${1:-$HOME}"

section() {
  printf '\n=== %s ===\n' "$1"
}

run_if_available() {
  local command_name="$1"
  shift
  if command -v "$command_name" >/dev/null 2>&1; then
    "$command_name" "$@" 2>&1
  else
    printf '%s: NOT_FOUND\n' "$command_name"
  fi
}

section "OS"
uname -a 2>&1
if [ -r /etc/os-release ]; then
  cat /etc/os-release
fi

section "CPU"
run_if_available lscpu

section "MEMORY"
run_if_available free -h

section "TARGET STORAGE"
printf 'requested_target=%s\n' "$target_path"
df -hT "$target_path" 2>&1 || true

section "NVIDIA"
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=index,name,memory.total,driver_version,compute_cap \
    --format=csv,noheader 2>&1
  printf '\nGPU topology:\n'
  nvidia-smi topo -m 2>&1 || true
else
  printf 'nvidia-smi: NOT_FOUND\n'
fi

section "PYTHON AND ENV MANAGERS"
run_if_available python3 --version
run_if_available conda --version
run_if_available micromamba --version
run_if_available mamba --version
run_if_available uv --version

section "CONTAINERS"
run_if_available docker --version
if command -v docker >/dev/null 2>&1; then
  docker info --format \
    'server={{.ServerVersion}} rootless={{.SecurityOptions}} runtime={{.DefaultRuntime}}' \
    2>&1 || true
fi
run_if_available podman --version

section "BUILD AND TRANSFER TOOLS"
run_if_available gcc --version
run_if_available git --version
run_if_available tar --version
run_if_available zstd --version
run_if_available rsync --version
run_if_available sha256sum --version

section "NETWORK REACHABILITY"
if command -v curl >/dev/null 2>&1; then
  curl -I --max-time 10 https://huggingface.co 2>&1 | head -n 8
else
  printf 'curl: NOT_FOUND\n'
fi

section "CURRENT USER AND PERMISSIONS"
id 2>&1
if [ -d "$target_path" ] && [ -w "$target_path" ]; then
  printf 'target_writable=true\n'
else
  printf 'target_writable=false\n'
fi

section "NOTES"
printf '%s\n' \
  'This report intentionally excludes passwords, SSH private keys, API tokens, and model-service secrets.'
