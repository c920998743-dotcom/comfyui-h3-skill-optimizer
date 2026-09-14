# H3 本地 24GB 版（待 RunningHub 实机验证）

选定模型：**Qwen/Qwen2.5-Omni-7B-AWQ**。本版在运行 ComfyUI 的机器上直接加载模型，不调用 Gemini、OpenAI 或其他付费模型 API，不需要 API Key。推理阶段强制离线，模型必须预先下载完整。

## 先确认 RunningHub 的部署条件

本包是自定义节点源码和工作流，不是 RunningHub 已上架应用。**上传 JSON 不会自动安装 Python 节点、模型或独立环境。**目前尚未核实你账号所在实例是否允许这些操作：

- 安装本包的 ComfyUI 自定义节点及前端扩展；
- 建立可执行的独立 Python 虚拟环境，安装依赖并启动子进程；
- 存放、读取完整 AWQ 模型目录；
- 使用 CUDA 并释放之前加载的模型。

若平台只有工作流编辑权限，需要通过 RunningHub 支持的节点安装/提交渠道或平台工作人员完成环境部署。不要直接在现有 H3 Python 环境降级 Transformers。节点缺失或环境不存在时，这份 JSON 不能单独运行。

## 交付内容

- `MiniMax-H3-本地24G-Skill优化.json`：修改后的工作流。
- `comfyui_h3_local_optimizer/`：完整节点目录，含 H3 skill、worker、预览和环境检查。
- `tests/`：离线测试。模型权重未打包。

将完整节点目录放在 `ComfyUI/custom_nodes/`，安装其中 `requirements.txt` 并重启 ComfyUI。这个 requirements 仅包含轻量节点依赖。

## 独立模型环境

由有实例安装权限的人执行；选择实例中允许写入且能持久保存的目录：

```bash
bash ComfyUI/custom_nodes/comfyui_h3_local_optimizer/install_worker.sh /your/writable/path/h3-omni-venv
```

脚本通过系统 `python3 -m venv` 建立完全独立环境，使用 Torch 2.6.0、Transformers 4.52.3 和 AutoAWQ 0.2.9。运行实例须有与此 CUDA 构建兼容的驱动、Linux/NVIDIA GPU。依赖包是否能在 RunningHub 当前镜像成功安装尚未验证；不能用“环境检查成功”代替真实模型加载测试。

该版本组合来自官方 AWQ 使用建议并参考源码适配。AutoAWQ 是旧依赖，因此采用子进程隔离，不覆盖 H3 的 Python 包。普通 AWQ 加载路线与官方特制 low-VRAM 演示并非同一实现，不能照搬其显存数字作为本节点保证。

## 模型准备

下载[官方 Qwen2.5-Omni-7B-AWQ](https://huggingface.co/Qwen/Qwen2.5-Omni-7B-AWQ)的**完整仓库快照**，包括所有 safetensors 分片、索引、配置、处理器、tokenizer 和 `spk_dict.pt`。不得只上传一个量化权重文件。可在具备下载环境的机器准备后，通过平台允许的模型上传方式部署。

示例位置：`ComfyUI/models/LLM/Qwen2.5-Omni-7B-AWQ/`。

然后用独立 Python 运行预检查：

```bash
/your/writable/path/h3-omni-venv/bin/python \
  ComfyUI/custom_nodes/comfyui_h3_local_optimizer/check_environment.py \
  --model /absolute/path/ComfyUI/models/LLM/Qwen2.5-Omni-7B-AWQ
```

预检查只检查依赖导入、CUDA 和必要文件，不加载大模型，不执行推理。通过后还需实际运行一次优化任务。

## 节点配置

导入工作流，在 **323：H3 · 本地24G Skill优化** 中设置：

| 参数 | 默认/建议 |
|---|---|
| `worker_python` | 改成真实独立环境 Python 绝对路径；默认 `/opt/h3-omni-venv/bin/python` 只是常见路径示例，安装脚本不会自动创建它 |
| `model_path` | `LLM/Qwen2.5-Omni-7B-AWQ`，相对于 ComfyUI/models；也支持绝对目录 |
| `skill_path` | 留空读取节点内置完整 H3 skill |
| `image_side` | 448，分析图像长边上限，不改变 H3 生成分辨率 |
| `video_frames` | 8，覆盖 H3 有效参考时段的均匀采样，包含开头和结尾 |
| `audio_chunk_seconds` | 6，分块覆盖全部音频，不只听前6秒 |
| `max_new_tokens` | 3072，最终提示词输出上限 |
| `timeout_seconds` | 1800，本地加载和多次推理可能较慢 |
| `refresh` | 增大数字触发重新优化，输入不变时遵循 ComfyUI 缓存 |

模型加载时关闭语音输出组件，仅用 Thinker 生成文字。AWQ 浮点排除列表同时保护视觉、音频编码器与 lm_head，匹配官方权重结构。

## 24GB 执行流程与实际限制

1. 按原素材启用状态接收图片、视频、音频；现有图片1–3启用，其余组仍绕过。
2. 卸载 ComfyUI 管理的已有 GPU 模型。
3. 独立进程加载 AWQ；每张图片、每个视频采样序列、每段音频分别分析。
4. 视觉分析使用 JPEG，视频是**带原时间戳的采样图片序列**，并非原生完整视频理解；高速动作、短暂字幕和精确运镜可能遗漏。
5. 音频完整分块后重采样为16k单声道用于理解；原始音频仍送H3。分块边界可能影响连贯台词识别，要求不清楚处标记 `[unclear]`。
6. 将全部分析笔记、原始提示词、引用映射和完整 skill 汇总，再生成六段 H3 提示词。最终整合通过**分析笔记**了解素材，不会在同一次推理中重读所有媒体；复杂跨素材关系可能有信息损失。
7. 子进程退出，释放模型和 CUDA 上下文；正常返回、失败或用户取消均等待子进程结束。
8. H3 模型加载节点才开始执行，随后进行原一采/二采生成。

本版是24GB容量目标方案，**没有实测峰值，不保证任何数量素材都能在24GB运行**。完整规则不静默截断；上下文超预算直接报错。显存不足时先减小 `image_side`、`video_frames` 或单次参考素材数量，保留所需内容后再验证。worker成功后预览会显示其PyTorch峰值分配显存，未包含父进程和CUDA全部开销。

## 新增节点及使用位置

- **323 本地优化节点**：替换原API优化节点，无密钥字段。
- **324 本地优化结果**：显示最终提示词、引用及素材分析笔记。
- **326–329 优化完成后加载**：四个轻量顺序节点，分别控制36/37 VAE、38 CLIP、192 UNET。这些不是第二次模型优化，不额外运行LLM。

**修改 H3 的四个基础模型文件名时，要改326–329顺序节点里的 value**，因为原加载器的文件名已被外部连线接管。LoRA和采样参数继续在原节点调整。

只测试提示词时，禁用（Mute）原视频输出214和264，保留324；先用一张图片＋一句中文需求跑通，再逐步加入音频和视频。确认正常后恢复视频输出。不要把四个顺序节点绕过，否则可能提前加载 H3 占满显存。

## 当前验证状态

已验证素材分块覆盖、时间戳、有效引用编号、完整skill传递、AWQ配置、子进程离线标志与取消清理、125条工作流连线及四个加载顺序依赖。测试用合成素材和模拟推理，没有真实 AWQ 模型结果。

**未验证：RunningHub 节点安装、独立环境依赖安装、真实 AWQ 推理、24GB峰值显存、最终 H3 视频生成。**这是一份可供部署验证的本地版，不是已经在 RunningHub 跑通的应用。

参考：[官方AWQ说明](https://huggingface.co/Qwen/Qwen2.5-Omni-7B-AWQ)、[Qwen2.5-Omni官方代码](https://github.com/QwenLM/Qwen2.5-Omni)、[Transformers 4.52.3 Omni实现](https://github.com/huggingface/transformers/blob/v4.52.3/src/transformers/models/qwen2_5_omni/modeling_qwen2_5_omni.py)。
