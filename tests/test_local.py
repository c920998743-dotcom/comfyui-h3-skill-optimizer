import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from comfyui_h3_local_optimizer.media import export_job
from comfyui_h3_local_optimizer.worker import optimize_job, local_config
from comfyui_h3_local_optimizer.nodes import H3AfterPrompt


class LocalTests(unittest.TestCase):
    def test_export_uses_every_audio_chunk_and_spread_video_frames(self):
        audio = {'waveform': np.zeros((1, 1, 16000 * 13)), 'sample_rate': 16000}
        media = {'image_2': np.zeros((1, 16, 16, 3)),
                 'video_2': np.zeros((360, 16, 16, 3)), 'video_audio_2': audio, 'audio_3': audio}
        with tempfile.TemporaryDirectory() as tmp:
            job = export_job(tmp, '参考动作，音频只用音色。', 362, 15, 544, 960, '', media, 448, 8, 6)
            self.assertEqual([r['label'] for r in job['references']],
                             ['<Picture 1>', '<Audio 1>', '<Video 1>', '<Audio 2>'])
            clips = [x for x in job['assets'] if x['label'] == '<Audio 2>']
            self.assertEqual(len(clips), 3)
            self.assertEqual([(c['start'],c['end']) for c in clips], [(0,6),(6,12),(12,13)])
            video = next(x for x in job['assets'] if x['kind'] == 'video')
            self.assertEqual(len(video['paths']), 8)
            self.assertEqual(video['timestamps'][-1], 344 / 24)
            self.assertTrue(all(Path(p).exists() for p in video['paths']))
            self.assertAlmostEqual(job['effective_duration'], 362/24)

    def test_no_media_retains_full_rules(self):
        with tempfile.TemporaryDirectory() as tmp:
            job = export_job(tmp, '一个安静的房间', 362, 15, 544, 960, '', {}, 448, 8, 6)
            self.assertEqual(job['assets'], [])
            self.assertIn('integrated_multimodal_description', job['rules'])

    def test_invalid_frames_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):
                export_job(tmp, 'x', 360, 15, 544, 960, '', {}, 448, 8, 6)

    def test_every_asset_is_analyzed_before_final_rewrite(self):
        calls = []
        def infer(prompt, asset, tokens):
            calls.append((prompt,asset))
            return 'observed facts' if asset else 'final result'
        with tempfile.TemporaryDirectory() as tmp:
            job = export_job(tmp, '台词保持你好', 362, 15, 544, 960, '',
                             {'image_1': np.zeros((1,16,16,3))},448,8,6)
            result = optimize_job(job,infer,3072)
        self.assertEqual(len(calls),2)
        self.assertIsNone(calls[-1][1])
        self.assertIn('observed facts',calls[-1][0])
        self.assertIn('Full-Reference Mode Rewrite Output Format Guide', calls[-1][0])
        self.assertEqual(result['prompt'],'final result')

    def test_awq_skip_list_matches_float_components(self):
        from types import SimpleNamespace
        config = SimpleNamespace(quantization_config={'quant_method':'awq','bits':4})
        local_config(config)
        self.assertFalse(config.enable_audio_output)
        self.assertIn('audio_tower',config.quantization_config['modules_to_not_convert'])

    def test_gate_needs_successful_prompt(self):
        with self.assertRaises(ValueError):
            H3AfterPrompt().release('', 'model.safetensors')
        self.assertEqual(H3AfterPrompt().release('ready','model.safetensors'),('model.safetensors',))


class WorkflowTests(unittest.TestCase):
    def test_local_workflow_has_no_api_and_all_loaders_are_gated(self):
        root=Path(__file__).resolve().parents[1]
        d=json.loads((root/'MiniMax-H3-本地24G-Skill优化.json').read_text(encoding='utf-8'))
        nodes={n['id']:n for n in d['nodes']}
        links={v[0]:v for v in d['links']}
        self.assertEqual(nodes[323]['type'],'H3LocalSkillOptimizer')
        self.assertNotIn('api_key_env',[i['name'] for i in nodes[323]['inputs']])
        for lid,src,out,dst,slot,kind in links.values():
            self.assertIn(lid,nodes[src]['outputs'][out]['links'])
            self.assertEqual(nodes[dst]['inputs'][slot]['link'],lid)
        for nid in [36,37,38,192]:
            incoming=next(i for i in nodes[nid]['inputs'] if i.get('link'))
            gate=nodes[links[incoming['link']][1]]
            self.assertEqual(gate['type'],'H3AfterPrompt')
            self.assertEqual(links[gate['inputs'][0]['link']][1],323)


if __name__ == '__main__':
    unittest.main()
