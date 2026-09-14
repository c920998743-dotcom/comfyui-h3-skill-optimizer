import json
from pathlib import Path
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from comfyui_h3_local_optimizer.nodes import H3LocalSkillOptimizer


class ProcessTests(unittest.TestCase):
    def test_worker_exits_before_result_return_and_uses_offline_environment(self):
        events=[]
        folder=ModuleType('folder_paths')
        comfy=ModuleType('comfy')
        mm=ModuleType('comfy.model_management')
        mm.unload_all_models=lambda:events.append('unload')
        mm.soft_empty_cache=lambda:events.append('empty')
        mm.get_torch_device=lambda:SimpleNamespace(type='cuda',index=0)
        mm.throw_exception_if_processing_interrupted=lambda:None
        comfy.model_management=mm

        class Process:
            returncode=0
            def __init__(self, command, **kwargs):
                self.assert_offline=kwargs['env']['HF_HUB_OFFLINE']=='1'
                events.append('offline' if self.assert_offline else 'online')
                jobpath=Path(command[command.index('--job')+1])
                jobpath.with_name('result.json').write_text(json.dumps({'prompt':
                    'integrated_multimodal_description: [Shot 1] A quiet room.\n'
                    'overall_soundscape: Silence.\nnon_diegetic_music: N/A.',
                    'observations':[]}),encoding='utf-8')
            def poll(self):return 0
            def wait(self):events.append('exited');return 0

        with tempfile.TemporaryDirectory() as tmp:
            folder.models_dir=tmp
            model=Path(tmp)/'model';model.mkdir()
            (model/'config.json').write_text('{}')
            with patch.dict(sys.modules,{'folder_paths':folder,'comfy':comfy,'comfy.model_management':mm}),\
                 patch('comfyui_h3_local_optimizer.nodes.subprocess.Popen',Process):
                result=H3LocalSkillOptimizer().optimize('安静房间',15,362,544,960,sys.executable,str(model))
        self.assertEqual(events,['unload','empty','offline','exited'])
        self.assertIn('quiet room',result['result'][0])

    def test_interruption_kills_and_waits_for_worker(self):
        events=[]
        folder=ModuleType('folder_paths');comfy=ModuleType('comfy');mm=ModuleType('comfy.model_management')
        mm.unload_all_models=lambda:None
        mm.soft_empty_cache=lambda:None
        mm.get_torch_device=lambda:SimpleNamespace(type='cuda',index=0)
        def interrupt():raise RuntimeError('user cancelled')
        mm.throw_exception_if_processing_interrupted=interrupt
        comfy.model_management=mm
        class Process:
            def __init__(self,*a,**kw):pass
            def poll(self):return None
            def kill(self):events.append('kill')
            def wait(self):events.append('wait')
        with tempfile.TemporaryDirectory() as tmp:
            folder.models_dir=tmp
            (Path(tmp)/'config.json').write_text('{}')
            with patch.dict(sys.modules,{'folder_paths':folder,'comfy':comfy,'comfy.model_management':mm}),\
                 patch('comfyui_h3_local_optimizer.nodes.subprocess.Popen',Process):
                with self.assertRaisesRegex(RuntimeError,'user cancelled'):
                    H3LocalSkillOptimizer().optimize('x',15,362,544,960,sys.executable,tmp)
        self.assertEqual(events,['kill','wait'])


if __name__=='__main__':unittest.main()
