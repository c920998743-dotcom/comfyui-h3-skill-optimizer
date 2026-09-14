import json
from pathlib import Path
from comfyui_h3_local_optimizer.nodes import H3LocalSkillOptimizer

ROOT = Path(__file__).resolve().parent
source = ROOT.parent / 'H3智能提示词工作流/MiniMax-H3-多参考生视频-Skill智能优化.json'
d = json.loads(source.read_text(encoding='utf-8'))
nodes = {n['id']: n for n in d['nodes']}
n = nodes[323]
old_inputs = n['inputs']
connections = {i['name']: i.get('link') for i in old_inputs}
inputs, widgets = [], []
for group, fields in H3LocalSkillOptimizer.INPUT_TYPES().items():
    for name, spec in fields.items():
        options = spec[1] if len(spec) > 1 else {}
        item = {'name': name, 'type': spec[0], 'link': connections.get(name)}
        if group == 'optional':
            item['shape'] = 7
        elif not options.get('forceInput'):
            item['widget'] = {'name': name}
            widgets.append(options['default'])
        inputs.append(item)
for link in d['links']:
    if link[3] == 323:
        name = old_inputs[link[4]]['name']
        link[4] = next(i for i, item in enumerate(inputs) if item['name'] == name)
n.update(type='H3LocalSkillOptimizer', title='H3 · 本地24G Skill优化', inputs=inputs,
         widgets_values=widgets, properties={'Node name for S&R': 'H3LocalSkillOptimizer'})
nodes[324].update(type='H3LocalPromptPreview', title='H3 · 本地优化结果',
                  properties={'Node name for S&R': 'H3LocalPromptPreview'})
nodes[325]['widgets_values'] = ['''24GB本地版：假设RunningHub已提供本地Qwen2.5-Omni-7B-AWQ及依赖，不填写路径和接口地址。
运行前必须安装独立Omni环境及完整Qwen2.5-Omni-7B-AWQ模型。
model_name只选择已提供的本地模型名称，节点内部自动查找。
图片默认448，视频默认均匀抽8帧（可调），音频按6秒分段完整分析。
模型逐素材分析后读取完整H3 skill生成提示词；退出子进程释放显存。
右侧4个顺序节点保存原模型文件名；要更换H3模型，请在对应顺序节点修改。
这些顺序节点保证优化完成后才开始加载H3模型/CLIP/VAE，不要绕过。
本包未经过RunningHub实机验证；只有平台能安装自定义节点、独立Python依赖和模型时才可运行。
仅测试优化时禁用原214/264输出节点，保留324预览。
''']
lid = max(x[0] for x in d['links']) + 1
nid = max(nodes) + 1
for j, loader_id in enumerate([36, 37, 38, 192]):
    loader = nodes[loader_id]
    field = {'VAELoader': 'vae_name', 'CLIPLoader': 'clip_name', 'UNETLoader': 'unet_name'}[loader['type']]
    slot = next(i for i, v in enumerate(loader['inputs']) if v['name'] == field)
    gate = {'id': nid, 'type': 'H3AfterPrompt', 'title': '优化完成后加载 · ' + loader['type'] + f' #{loader_id}',
            'pos': [7180, 2300+j*150], 'size': [500,110], 'mode': 0, 'flags': {},
            'order': len(d['nodes']), 'inputs': [
                {'name': 'prompt_ready', 'type': 'STRING', 'link': lid},
                {'name': 'value', 'type': 'STRING', 'widget': {'name': 'value'}, 'link': None}],
            'outputs': [{'name': 'value', 'type': '*', 'links': [lid+1]}],
            'widgets_values': [loader['widgets_values'][0]],
            'properties': {'Node name for S&R': 'H3AfterPrompt'}}
    d['nodes'].append(gate)
    n['outputs'][0]['links'].append(lid)
    loader['inputs'][slot]['link'] = lid+1
    d['links'].extend([[lid,323,0,nid,0,'STRING'],[lid+1,nid,0,loader_id,slot,'*']])
    nid += 1
    lid += 2
d['last_node_id'] = nid-1
d['last_link_id'] = lid-1
d['revision'] += 1
d['groups'][-1]['title'] = '24GB · 本地Skill优化与模型加载顺序'
d['groups'][-1]['bounding'][2] = 2500
target=ROOT/'MiniMax-H3-本地24G-Skill优化.json'
target.write_text(json.dumps(d,ensure_ascii=False,indent=2),encoding='utf-8')
print(len(d['nodes']), 'nodes', len(d['links']), 'links')
