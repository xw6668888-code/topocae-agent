#!/usr/bin/env bash

set -u

target_path="${1:-$HOME}"

section() {
  printf '\n=== %s ===\n' "$1"
}

probe_url() {
  local label="$1"
  local url="$2"
  printf '%s\n' "-- $label: $url"
  if command -v curl >/dev/null 2>&1; then
    curl -L -I --connect-timeout 5 --max-time 15 "$url" 2>&1 | head -n 12
  else
    printf 'curl: NOT_FOUND\n'
  fi
}

section "IDENTITY AND HOST"
printf 'user=%s\n' "$(id -un 2>/dev/null || true)"
printf 'host=%s\n' "$(hostname 2>/dev/null || true)"
printf 'home=%s\n' "$HOME"
printf 'target=%s\n' "$target_path"

section "TARGET DIRECTORY"
mkdir -p "$target_path" 2>/dev/null || true
df -hT "$target_path" 2>&1 || true
df -i "$target_path" 2>&1 || true
if command -v quota >/dev/null 2>&1; then
  quota -s 2>&1 || true
else
  printf 'quota: NOT_FOUND_OR_NOT_EXPOSED\n'
fi
if [ -w "$target_path" ]; then
  printf 'target_writable=true\n'
else
  printf 'target_writable=false\n'
fi

section "PRIVILEGES"
if command -v sudo >/dev/null 2>&1; then
  if sudo -n true >/dev/null 2>&1; then
    printf 'passwordless_sudo=true\n'
  else
    printf 'passwordless_sudo=false\n'
  fi
else
  printf 'sudo=NOT_FOUND\n'
fi
if groups 2>/dev/null | grep -Eq '(^|[[:space:]])docker([[:space:]]|$)'; then
  printf 'docker_group=true\n'
else
  printf 'docker_group=false\n'
fi

section "PYTHON BOOTSTRAP"
python3 --version 2>&1 || true
python3 -c "import venv; print('python_venv_module=true')" 2>&1 || true
python3 -m pip --version 2>&1 || true
command -v nvcc >/dev/null 2>&1 && nvcc --version 2>&1 || printf 'nvcc=NOT_FOUND\n'

section "DNS"
for host in pypi.org files.pythonhosted.org conda.anaconda.org repo.anaconda.com huggingface.co github.com; do
  printf '%-28s ' "$host"
  getent ahosts "$host" 2>&1 | head -n 1 || true
done

section "HTTPS ENDPOINTS"
probe_url "PyPI" "https://pypi.org/simple/pip/"
probe_url "Conda Forge" "https://conda.anaconda.org/conda-forge/linux-64/repodata.json.zst"
probe_url "Miniforge" "https://github.com/conda-forge/miniforge/releases/latest"
probe_url "Hugging Face" "https://huggingface.co/Qwen/Qwen3-8B"

section "GPU AVAILABILITY"
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=index,name,memory.total,memory.used,utilization.gpu,compute_mode \
    --format=csv,noheader 2>&1
  printf '\nActive compute processes:\n'
  nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory \
    --format=csv,noheader 2>&1 || true
else
  printf 'nvidia-smi=NOT_FOUND\n'
fi

section "TRANSFER TOOLS"
for command_name in ssh scp rsync tar zstd sha256sum; do
  if command -v "$command_name" >/dev/null 2>&1; then
    printf '%s=%s\n' "$command_name" "$(command -v "$command_name")"
  else
    printf '%s=NOT_FOUND\n' "$command_name"
  fi
done

section "NOTES"
printf '%s\n' \
  'No installation or system modification was attempted.' \
  'Do not append passwords, private keys, tokens, proxy credentials, or other secrets to this report.'
