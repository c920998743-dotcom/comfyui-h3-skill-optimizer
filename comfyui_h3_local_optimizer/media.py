"""Export bounded, local-only inference jobs; audio chunks cover the full track."""
import base64
import json
from pathlib import Path

import numpy as np

from .core import as_numpy, effective_video_frames, encode_audio, load_rules, reference_manifest, rgb_image


def export_job(directory, prompt, target_frames, duration_seconds, width, height,
               skill_path, media, image_side, video_frames, audio_chunk_seconds):
    if not prompt.strip():
        raise ValueError('请填写原始提示词。')
    if target_frames < 5 or (target_frames - 5) % 17:
        raise ValueError('target_frames 必须为 H3 有效帧数 5+17n。')
    if not 224 <= image_side <= 768 or not 2 <= video_frames <= 24 or not 1 <= audio_chunk_seconds <= 15:
        raise ValueError('分析尺寸、视频帧数或音频分块设置不在允许范围。')
    root = Path(directory)
    refs = reference_manifest(media)
    assets = []
    for ref in refs:
        value = media[ref['port']]
        info = dict(ref)
        if ref['kind'] == 'image':
            path = root / (ref['port'] + '.jpg')
            if not len(value):
                raise ValueError('参考图片为空。')
            rgb_image(value[0], image_side).save(path, quality=90)
            info['paths'] = [str(path)]
            assets.append(info)
        elif ref['kind'] == 'video':
            count = effective_video_frames(len(value), target_frames)
            indices = np.unique(np.linspace(0, count - 1, min(video_frames, count)).round().astype(int))
            info.update(paths=[], timestamps=[int(i) / 24 for i in indices],
                        used_frames=count, duration=count / 24)
            for i in indices:
                path = root / f'{ref["port"]}_{i}.jpg'
                rgb_image(value[int(i)], image_side).save(path, quality=90)
                info['paths'].append(str(path))
            assets.append(info)
        else:
            waveform = as_numpy(value['waveform'])
            rate = int(value['sample_rate'])
            if waveform.ndim != 3 or waveform.shape[0] != 1 or rate <= 0 or waveform.shape[-1] == 0:
                raise ValueError('音频必须为非空单批次 [1,C,N]，采样率需大于0。')
            step = int(rate * audio_chunk_seconds)
            for start in range(0, waveform.shape[-1], step):
                end = min(start + step, waveform.shape[-1])
                data = encode_audio({'waveform': waveform[..., start:end], 'sample_rate': rate})
                path = root / f'{ref["port"]}_{start}.wav'
                path.write_bytes(base64.b64decode(data['inlineData']['data']))
                assets.append({**info, 'paths': [str(path)], 'start': start / rate, 'end': end / rate})
    job = {'prompt': prompt, 'effective_duration': target_frames / 24,
           'requested_duration': duration_seconds, 'width': width, 'height': height,
           'references': refs, 'assets': assets, 'rules': load_rules(skill_path)}
    (root / 'job.json').write_text(json.dumps(job, ensure_ascii=False), encoding='utf-8')
    return job
