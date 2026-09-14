"""Install Omni in a private venv; never change the caller's site-packages."""
import os
from pathlib import Path
import subprocess
import venv


def main():
    root = Path(__file__).resolve().parent
    destination = root / '.omni-env'
    venv.EnvBuilder(with_pip=True, system_site_packages=False).create(destination)
    python = destination / ('Scripts/python.exe' if os.name == 'nt' else 'bin/python')
    env = dict(os.environ, PYTHONNOUSERSITE='1')
    for name in ('PYTHONPATH', 'PYTHONHOME', 'VIRTUAL_ENV'):
        env.pop(name, None)

    def run(*args):
        subprocess.run([str(python), '-I', *args], env=env, check=True)

    run('-m', 'pip', 'install', '--upgrade', 'pip')
    run('-m', 'pip', 'install', '--index-url', 'https://download.pytorch.org/whl/cu128',
        'torch==2.7.1', 'torchvision==0.22.1', 'torchaudio==2.7.1')
    run('-m', 'pip', 'install', '-r', str(root / 'worker-requirements.txt'))
    run('-m', 'pip', 'check')
    run('-c', 'import torch, awq; from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor; print("Omni runtime imports OK", torch.__version__)')
    print('专用环境安装完成；模型权重无需重新下载。')


if __name__ == '__main__':
    main()
