import os
from pathlib import Path
import unittest
from unittest.mock import patch

from comfyui_h3_local_optimizer.runtime import worker_environment, worker_python


class RuntimeTests(unittest.TestCase):
    def test_missing_private_runtime_does_not_use_shared_python(self):
        with patch.object(Path, 'is_file', return_value=False):
            with self.assertRaisesRegex(RuntimeError, 'install_runtime.py'):
                worker_python()

    def test_parent_python_paths_are_removed_but_cuda_selection_is_retained(self):
        with patch.dict(os.environ, {'PYTHONPATH': '/shared/packages',
                                    'PYTHONHOME': '/shared/python',
                                    'VIRTUAL_ENV': '/shared/env',
                                    'CUDA_VISIBLE_DEVICES': '0'}):
            env = worker_environment()
        for key in ('PYTHONPATH', 'PYTHONHOME', 'VIRTUAL_ENV'):
            self.assertNotIn(key, env)
        self.assertEqual(env['CUDA_VISIBLE_DEVICES'], '0')
        self.assertEqual(env['PYTHONNOUSERSITE'], '1')
