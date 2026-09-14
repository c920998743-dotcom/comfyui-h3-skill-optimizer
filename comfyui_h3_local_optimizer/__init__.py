from .nodes import H3LocalSkillOptimizer, H3LocalPromptPreview, H3AfterPrompt

NODE_CLASS_MAPPINGS = {'H3LocalSkillOptimizer': H3LocalSkillOptimizer,
                       'H3LocalPromptPreview': H3LocalPromptPreview,
                       'H3AfterPrompt': H3AfterPrompt}
NODE_DISPLAY_NAME_MAPPINGS = {'H3LocalSkillOptimizer': 'H3 · 本地24G Skill优化',
                              'H3LocalPromptPreview': 'H3 · 本地优化结果',
                              'H3AfterPrompt': 'H3 · 优化完成后加载'}
WEB_DIRECTORY = './web'
