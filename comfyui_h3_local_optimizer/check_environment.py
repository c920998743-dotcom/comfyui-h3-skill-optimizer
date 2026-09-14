"""No download, no model load, no API access. Run using worker Python."""
import argparse
import importlib.metadata
import json
from pathlib import Path


def check(path):
    import torch
    import awq
    from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor
    model = Path(path)
    for name in ['config.json', 'preprocessor_config.json', 'tokenizer_config.json', 'spk_dict.pt']:
        if not (model / name).is_file():
            raise ValueError('缺少模型文件：' + name)
    index = model / 'model.safetensors.index.json'
    if not index.is_file():
        raise ValueError('缺少AWQ模型权重索引。')
    manifest = json.loads(index.read_text())
    for filename in set(manifest['weight_map'].values()):
        if not (model / filename).is_file():
            raise ValueError('缺少权重分片：' + filename)
    if not torch.cuda.is_available():
        raise ValueError('当前worker环境无法使用CUDA。')
    config = json.loads((model / 'config.json').read_text())
    if config.get('quantization_config', {}).get('quant_method') != 'awq':
        raise ValueError('目录不是AWQ模型。')
    print(json.dumps({'status': 'PRECHECK_OK_NOT_INFERENCE_TEST',
        'torch': torch.__version__, 'transformers': importlib.metadata.version('transformers'),
        'autoawq': importlib.metadata.version('autoawq'),
        'cuda': torch.version.cuda, 'gpu': torch.cuda.get_device_name(),
        'gpu_total_gib': torch.cuda.get_device_properties(0).total_memory / 1024**3}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True)
    check(parser.parse_args().model)
