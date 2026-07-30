# Linux 环境恢复

本目录的规格由 node6 上实际通过验证的环境导出，不使用项目根目录的旧 `environment.yml`。

## Core 环境

```bash
/home/ywx1/miniforge3/bin/mamba env create \
  --file migration/linux/topocae-core.environment.yml \
  --override-channels
```

`topocae-core-linux-64.explicit.txt` 是 node6 `linux-64` 平台的精确 conda 锁定清单，可用于相同平台的逐包恢复。

## LLM 环境

vLLM 官方建议由 `uv` 自动选择与 NVIDIA 驱动匹配的 PyTorch wheel。为复现本次安装：

```bash
/home/ywx1/miniforge3/bin/mamba create -y -n topocae-llm \
  --override-channels -c conda-forge python=3.12 pip uv

/home/ywx1/miniforge3/envs/topocae-llm/bin/uv pip install \
  --python /home/ywx1/miniforge3/envs/topocae-llm/bin/python \
  --torch-backend=auto 'vllm==0.26.0' 'openai==2.48.0'
```

`topocae-llm-conda-linux-64.explicit.txt` 锁定基础 conda 环境，`topocae-llm-pip.freeze.txt` 锁定安装后的 Python 包。该环境只完成软件安装与 import/CLI 检查；没有下载模型权重，也没有启动服务或调用 LLM。
