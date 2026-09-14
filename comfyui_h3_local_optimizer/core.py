"""H3 rules, reference numbering and prompt validation. No model API calls."""
import base64
import hashlib
import io
from pathlib import Path
import re
import wave

import numpy as np
from PIL import Image

RULE_FILES = ('SKILL.md', 'references/base-en.txt', 'references/ref-en.txt')
REF_FIELDS = ('subject_definitions', 'summary', 'retention_analysis',
              'detailed_description', 'overall_soundscape', 'non_diegetic_music')
BASE_FIELDS = ('integrated_multimodal_description', 'overall_soundscape', 'non_diegetic_music')


def rule_root(path):
    return Path(path).expanduser() if path.strip() else Path(__file__).parent / 'skill'


def load_rules(path):
    root = rule_root(path)
    return '\n\n'.join('### ' + name + '\n' + (root / name).read_text(encoding='utf-8-sig')
                        for name in RULE_FILES)


def rules_digest(path):
    return hashlib.sha256(load_rules(path).encode('utf-8')).hexdigest()


def as_numpy(value):
    if hasattr(value, 'detach'):
        value = value.detach().float().cpu().numpy()
    return np.asarray(value)


def inline(data, mime):
    return {'inlineData': {'mimeType': mime, 'data': base64.b64encode(data).decode('ascii')}}


def rgb_image(frame, max_side):
    array = as_numpy(frame)
    if array.ndim != 3 or array.shape[-1] not in (3, 4):
        raise ValueError('IMAGE 必须是 [帧数, 高, 宽, RGB/RGBA] 格式。')
    if not np.isfinite(array).all():
        raise ValueError('图片包含非有限像素值。')
    result = Image.fromarray((np.clip(array[..., :3], 0, 1) * 255).astype(np.uint8))
    result.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
    return result


def encode_image(images, max_side):
    # H3 uses img[:1] for each reference-image port.
    if len(images) == 0:
        raise ValueError('参考图片为空。')
    image = rgb_image(images[0], max_side)
    data = io.BytesIO()
    image.save(data, 'JPEG', quality=90)
    return inline(data.getvalue(), 'image/jpeg')


def encode_audio(audio):
    array = as_numpy(audio['waveform'])
    rate = int(audio['sample_rate'])
    if array.ndim != 3 or array.shape[0] != 1 or array.shape[1] not in (1, 2):
        raise ValueError('AUDIO 需要单批次、单声道或双声道 [1, C, N]，不能静默丢弃音频批次。')
    if rate <= 0 or array.shape[2] == 0 or not np.isfinite(array).all():
        raise ValueError('音频为空或采样率/采样值无效。')
    pcm = (np.clip(array[0].T, -1, 1) * 32767).astype('<i2').tobytes()
    buf = io.BytesIO()
    with wave.open(buf, 'wb') as stream:
        stream.setnchannels(array.shape[1])
        stream.setsampwidth(2)
        stream.setframerate(rate)
        stream.writeframes(pcm)
    return inline(buf.getvalue(), 'audio/wav')


def effective_video_frames(count, target_frames):
    count = min(count, target_frames)
    if count < 5:
        raise ValueError('H3 参考视频至少需要 5 帧。')
    return count - ((count - 5) % 17)


def reference_manifest(media):
    refs = []
    image_id = video_id = audio_id = 0
    for i in range(1, 10):
        port = f'image_{i}'
        if media.get(port) is not None:
            image_id += 1
            refs.append({'port': port, 'label': f'<Picture {image_id}>', 'kind': 'image'})
    for i in range(1, 4):
        port = f'video_{i}'
        if media.get(port) is None:
            continue
        video_id += 1
        soundtrack = f'video_audio_{i}'
        if media.get(soundtrack) is not None:
            audio_id += 1
            refs.append({'port': soundtrack, 'label': f'<Audio {audio_id}>', 'kind': 'audio',
                         'paired_video': f'<Video {video_id}>'})
        refs.append({'port': port, 'label': f'<Video {video_id}>', 'kind': 'video'})
    for i in range(1, 4):
        port = f'audio_{i}'
        if media.get(port) is not None:
            audio_id += 1
            refs.append({'port': port, 'label': f'<Audio {audio_id}>', 'kind': 'audio'})
    return refs


def validate_prompt(text, labels, duration):
    fields = REF_FIELDS if labels else BASE_FIELDS
    matches = list(re.finditer(r'(?m)^(' + '|'.join(dict.fromkeys(REF_FIELDS + BASE_FIELDS)) + r')[ \t]*:', text))
    if [m.group(1) for m in matches] != list(fields):
        raise ValueError('模型输出的 H3 段落缺失、重复或顺序错误，请重新优化。')
    sections = {m.group(1): text[m.end():matches[i + 1].start() if i + 1 < len(matches) else len(text)].strip()
                for i, m in enumerate(matches)}
    if any(not content for content in sections.values()):
        raise ValueError('模型输出含空段落，请明确填写内容或 N/A。')
    unknown = set(re.findall(r'<(?:Picture|Video|Audio) \d+>', text)) - set(labels)
    if unknown:
        raise ValueError('模型使用了不存在的参考标签：' + ', '.join(sorted(unknown)))
    if labels:
        defined = set(re.findall(r'(?m)^[ \t]*(<Subject \d+>)', sections['subject_definitions']))
        undefined = set(re.findall(r'<Subject \d+>', text)) - defined
        if undefined:
            raise ValueError('模型使用了未定义主体：' + ', '.join(sorted(undefined)))
    body_name = 'detailed_description' if labels else 'integrated_multimodal_description'
    body = sections[body_name]
    shots = re.findall(r'\[Shot (\d+)\]', body)
    if not shots or [int(s) for s in shots] != list(range(1, len(shots) + 1)):
        raise ValueError('模型输出镜头编号缺失、重复或不连续。')
    previous = 0.0
    for number in range(2, len(shots) + 1):
        cut = re.search(r'\[Shot ' + str(number) + r'\] At (\d{2}):(\d{2})\.(\d{3}),', body)
        if not cut:
            raise ValueError('后续镜头必须使用 [Shot N] At MM:SS.mmm, 格式。')
        minute, second, ms = map(int, cut.groups())
        timestamp = minute * 60 + second + ms / 1000
        if second >= 60 or not previous < timestamp < duration:
            raise ValueError('镜头切换时间未递增或超出有效视频时长。')
        previous = timestamp
    return text


