import json
from pathlib import Path
import subprocess
import tempfile
import time

from .core import rules_digest, validate_prompt
from .media import export_job
from .runtime import worker_python, worker_environment


class H3LocalSkillOptimizer:
    @classmethod
    def INPUT_TYPES(cls):
        return {'required': {
            'prompt': ('STRING', {'forceInput': True}),
            'duration_seconds': ('FLOAT', {'forceInput': True, 'default': 15}),
            'target_frames': ('INT', {'forceInput': True, 'default': 362}),
            'width': ('INT', {'forceInput': True, 'default': 544}),
            'height': ('INT', {'forceInput': True, 'default': 960}),
            'model_name': (['Qwen2.5-Omni-7B-AWQ'], {'default': 'Qwen2.5-Omni-7B-AWQ',
                'tooltip': '选择RunningHub已提供的本地模型；节点不要求填写路径或接口地址。'}),
            'skill_path': ('STRING', {'default': ''}),
            'image_side': ('INT', {'default': 448, 'min': 224, 'max': 768, 'step': 28}),
            'video_frames': ('INT', {'default': 8, 'min': 2, 'max': 24}),
            'audio_chunk_seconds': ('INT', {'default': 6, 'min': 1, 'max': 15}),
            'max_new_tokens': ('INT', {'default': 3072, 'min': 1024, 'max': 4096}),
            'timeout_seconds': ('INT', {'default': 1800, 'min': 60, 'max': 7200}),
            'refresh': ('INT', {'default': 0, 'min': 0, 'max': 2147483647}),
        }, 'optional': {
            **{f'image_{i}': ('IMAGE',) for i in range(1, 10)},
            **{f'video_{i}': ('IMAGE',) for i in range(1, 4)},
            **{f'video_audio_{i}': ('AUDIO',) for i in range(1, 4)},
            **{f'audio_{i}': ('AUDIO',) for i in range(1, 4)},
        }}

    RETURN_TYPES = ('STRING', 'STRING')
    RETURN_NAMES = ('优化后提示词', '引用映射与分析')
    FUNCTION = 'optimize'
    CATEGORY = 'H3/本地提示词助手'

    @classmethod
    def IS_CHANGED(cls, skill_path='', **kwargs):
        return rules_digest(skill_path)

    def optimize(self, prompt, duration_seconds, target_frames, width, height,
                 model_name, skill_path='', image_side=448,
                 video_frames=8, audio_chunk_seconds=6, max_new_tokens=3072,
                 timeout_seconds=1800, refresh=0, **media):
        import folder_paths
        import comfy.model_management as mm
        python = worker_python()
        resolver = getattr(folder_paths, 'get_full_path', None)
        model_value = (resolver('LLM', model_name) if resolver else None) or (resolver('checkpoints', model_name) if resolver else None) or str(Path(folder_paths.models_dir) / 'LLM' / model_name)
        model = Path(model_value)
        if not (model / 'config.json').is_file():
            raise ValueError('找不到所选 AWQ 模型，请将完整模型目录放入 ComfyUI/models/LLM/' + model_name + '。')
        # Release Comfy-managed GPU models before starting the separate CUDA process.
        mm.unload_all_models()
        mm.soft_empty_cache()
        with tempfile.TemporaryDirectory(prefix='h3_local_') as tmp:
            job = export_job(tmp, prompt, target_frames, duration_seconds, width, height,
                             skill_path, media, image_side, video_frames, audio_chunk_seconds)
            command = [str(python), '-I', str(Path(__file__).with_name('worker.py')),
                       '--job', str(Path(tmp) / 'job.json'), '--model', str(model),
                       '--max-new-tokens', str(max_new_tokens)]
            env = worker_environment()
            # Comfy CUDA device selection must also apply to the isolated process.
            device = mm.get_torch_device()
            if getattr(device, 'type', '') != 'cuda':
                raise ValueError('此24G版需要NVIDIA CUDA。')
            env['H3_OMNI_DEVICE'] = str(device.index or 0)
            with open(Path(tmp) / 'worker.log', 'w', encoding='utf-8') as log:
                proc = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, env=env,
                    creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
                started = time.monotonic()
                try:
                    while proc.poll() is None:
                        mm.throw_exception_if_processing_interrupted()
                        if time.monotonic() - started > timeout_seconds:
                            raise TimeoutError('本地优化超时，已终止Omni进程并释放其显存。')
                        time.sleep(0.2)
                finally:
                    if proc.poll() is None:
                        proc.kill()
                    proc.wait()
            result_file = Path(tmp) / 'result.json'
            if proc.returncode != 0 or not result_file.is_file():
                tail = (Path(tmp) / 'worker.log').read_text(encoding='utf-8', errors='replace')[-5000:]
                raise RuntimeError('本地Omni优化失败（未调用外部API）：\n' + tail)
            result = json.loads(result_file.read_text(encoding='utf-8'))
            text = validate_prompt(result['prompt'], [r['label'] for r in job['references']], target_frames / 24)
            mapping = json.dumps({'references': job['references'], 'observations': result['observations'],
                                 'peak_cuda_gib': result.get('peak_cuda_gib')}, ensure_ascii=False, indent=2)
        return {'ui': {'text': [text], 'mapping': [mapping]}, 'result': (text, mapping)}


class H3LocalPromptPreview:
    @classmethod
    def INPUT_TYPES(cls):
        return {'required': {'text': ('STRING', {'forceInput': True})},
                'optional': {'mapping': ('STRING', {'forceInput': True})}}

    RETURN_TYPES = ('STRING',)
    OUTPUT_NODE = True
    FUNCTION = 'show'
    CATEGORY = 'H3/本地提示词助手'

    def show(self, text, mapping=''):
        return {'ui': {'text': [text], 'mapping': [mapping]}, 'result': (text,)}


class H3AfterPrompt:
    """A string dependency gates filename inputs to H3's heavy loaders."""
    @classmethod
    def INPUT_TYPES(cls):
        return {'required': {'prompt_ready': ('STRING', {'forceInput': True}),
                             'value': ('STRING', {'default': ''})}}

    RETURN_TYPES = ('*',)
    FUNCTION = 'release'
    CATEGORY = 'H3/本地提示词助手'

    def release(self, prompt_ready, value):
        if not prompt_ready.strip():
            raise ValueError('必须先完成本地提示词优化。')
        return (value,)
