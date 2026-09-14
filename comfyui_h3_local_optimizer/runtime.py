"""Locate the node-owned runtime without falling back to ComfyUI's packages."""
import os
from pathlib import Path


def worker_python():
    root = Path(__file__).resolve().parent
    python = root / '.omni-env' / ('Scripts/python.exe' if os.name == 'nt' else 'bin/python')
    if not python.is_file():
        raise RuntimeError('Omni 专用环境未安装。请在节点目录运行 python install_runtime.py，然后重试。')
    return python


def worker_environment():
    env = dict(os.environ)
    for name in ('PYTHONPATH', 'PYTHONHOME', 'VIRTUAL_ENV'):
        env.pop(name, None)
    env.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
               TOKENIZERS_PARALLELISM='false', PYTHONIOENCODING='utf-8',
               PYTHONNOUSERSITE='1')
    return env
