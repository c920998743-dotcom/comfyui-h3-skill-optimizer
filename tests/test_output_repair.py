import unittest
from comfyui_h3_local_optimizer.worker import optimize_job, active_rules, timed_segments
from comfyui_h3_local_optimizer.core import load_rules, REF_FIELDS, BASE_FIELDS

SECTIONS = dict(subject_definitions='<Subject 1> from <Picture 1>', summary='Room',
                retention_analysis='Preserve', detailed_description='[Shot 1] A room. 你好',
                overall_soundscape='Silence.', non_diegetic_music='N/A.',
                integrated_multimodal_description='[Shot 1] A room. 你好')


def field(prompt):
    return prompt.split('H3 field ', 1)[1].split('.', 1)[0]


class OutputRepairTests(unittest.TestCase):
    def job(self):
        return dict(assets=[], prompt='保留台词“你好”', references=[{'label': '<Picture 1>'}],
                    requested_duration=15, effective_duration=362/24,
                    width=544, height=960, rules='Guide examples')

    def test_fields_always_assembled_in_h3_order(self):
        calls = []
        def infer(prompt, *args):
            calls.append(field(prompt))
            return SECTIONS[field(prompt)]
        result = optimize_job(self.job(), infer, 3072)['prompt']
        self.assertEqual(calls, list(REF_FIELDS))
        self.assertEqual(result, '\n\n'.join(k + ':\n' + SECTIONS[k] for k in REF_FIELDS))

    def test_translated_literal_triggers_one_repair(self):
        calls = []
        def infer(prompt, *args):
            calls.append(prompt)
            text = SECTIONS[field(prompt)]
            return text.replace('你好', 'Hello') if len(calls) <= 6 else text
        result = optimize_job(self.job(), infer, 3072)['prompt']
        self.assertIn('你好', result)
        self.assertEqual(len(calls), 12)
        self.assertIn('Previous validation failed:', calls[-1])

    def test_invalid_repair_is_not_returned_to_h3(self):
        calls = []
        def infer(prompt, *args):
            calls.append(prompt)
            return SECTIONS[field(prompt)].replace('<Picture 1>', '<Audio 1>')
        with self.assertRaisesRegex(ValueError, '不存在的参考'):
            optimize_job(self.job(), infer, 3072)
        self.assertEqual(len(calls), 12)

    def test_duplicate_heading_is_rejected(self):
        def infer(prompt, *args):
            return SECTIONS[field(prompt)] + '\nsummary: duplicate'
        with self.assertRaisesRegex(ValueError, '段落缺失、重复或顺序错误'):
            optimize_job(self.job(), infer, 3072)

    def test_quotes_only_in_summary_do_not_count_as_dialogue(self):
        sections = dict(SECTIONS, summary='你好', detailed_description='[Shot 1] A room.')
        with self.assertRaisesRegex(ValueError, '必须逐字保留'):
            optimize_job(self.job(), lambda prompt, *args: sections[field(prompt)], 3072)

    def test_no_reference_uses_three_fields(self):
        calls = []
        def infer(prompt, *args):
            calls.append(field(prompt))
            return SECTIONS[field(prompt)]
        optimize_job(dict(self.job(), references=[]), infer, 3072)
        self.assertEqual(calls, list(BASE_FIELDS))

    def test_mode_selection_excludes_complete_examples(self):
        rules = load_rules('')
        ref = active_rules(rules, True)
        base = active_rules(rules, False)
        self.assertIn('## 4.', ref)
        self.assertNotIn('## 7. Complete Example', ref)
        self.assertNotIn('## 5. Cases', base)
        self.assertNotIn('## 2. Final Prompt Structure', ref)
        self.assertNotIn('Full-Reference Mode', base)
        self.assertNotIn('```', ref)

    def test_explicit_timeline_preserves_nonuniform_cuts(self):
        job = dict(self.job(), prompt='0-3秒走近；3-6秒开盖；6-10秒上妆；10-13秒微笑；13-15秒展示。')
        calls = []
        def infer(prompt, *args):
            calls.append(prompt)
            return 'The woman moves.' if '仅改写这个镜头' in prompt else SECTIONS[field(prompt)]
        result = optimize_job(job, infer, 3072)['prompt']
        self.assertIn('[Shot 4] At 00:10.000,', result)
        self.assertIn('[Shot 5] At 00:13.000,', result)
        self.assertNotIn('[Shot 6]', result)
        self.assertEqual(sum('仅改写这个镜头' in p for p in calls), 5)

    def test_ambiguous_or_overlong_timeline_is_not_silently_rewritten(self):
        self.assertEqual(timed_segments('1-3秒开始；3-5秒结束', 15), [])
        self.assertEqual(timed_segments('0-3秒开始；4-5秒结束', 15), [])
        self.assertEqual(timed_segments('0-16秒结束', 15), [])

    def test_literal_placeholders_restore_exact_user_words(self):
        calls = []
        def infer(prompt, *args):
            calls.append(prompt)
            return SECTIONS[field(prompt)].replace('你好', 'H3_LITERAL_1')
        result = optimize_job(self.job(), infer, 3072)['prompt']
        self.assertIn('你好', result)
        self.assertNotIn('H3_LITERAL_', result)
        self.assertTrue(all('H3_LITERAL_1' in p for p in calls))

    def test_explicit_speech_and_subtitle_are_preserved_without_model_rewrite(self):
        job = dict(self.job(), prompt='0-5秒女生微笑，台词“你好”，字幕“欢迎”。')
        def infer(prompt, *args):
            if '仅改写这个镜头' in prompt:
                self.assertNotIn('你好', prompt)
                return 'The woman smiles.'
            return SECTIONS[field(prompt)]
        result = optimize_job(job, infer, 3072)['prompt']
        self.assertIn('<d>[Chinese] 你好</d>', result)
        self.assertIn('On-screen text reads "欢迎"', result)

    def test_markdown_subject_definition_preserves_real_identity(self):
        def infer(prompt, *args):
            return '- **<Subject 1>** from <Picture 1>' if field(prompt) == 'subject_definitions' else SECTIONS[field(prompt)]
        result = optimize_job(self.job(), infer, 3072)['prompt']
        self.assertIn('subject_definitions:\n<Subject 1> from <Picture 1>', result)

    def test_explicit_single_shot_gets_missing_first_marker(self):
        job = dict(self.job(), references=[], prompt='固定镜头，一个安静的房间。')
        result = optimize_job(job, lambda prompt, *args: SECTIONS[field(prompt)].replace('[Shot 1] ', ''), 3072)['prompt']
        self.assertIn('[Shot 1]', result)

    def test_literal_ten_is_not_replaced_as_literal_one(self):
        words = ['词' + str(i) for i in range(1, 11)]
        job = dict(self.job(), prompt=' '.join('“' + w + '”' for w in words))
        def infer(prompt, *args):
            if field(prompt) == 'detailed_description':
                return '[Shot 1] ' + ' '.join('H3_LITERAL_' + str(i) for i in range(1, 11))
            return SECTIONS[field(prompt)]
        result = optimize_job(job, infer, 3072)['prompt']
        self.assertIn('词10', result)
        self.assertNotIn('H3_LITERAL_', result)
