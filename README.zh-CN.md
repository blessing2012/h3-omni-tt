# H3 / LightX2V Turbo 源码交付

这是给 Tenstorrent 做适配的 NVIDIA 参考实现，具体组合为：

- MiniMax-H3 原版 `FL2VA` 基础模型。
- LightX2V `minimax_h3_fl2v_turbo_4step_v1.2_768p_bf16.safetensors`。
- vLLM-Omni 固定版本，加上 Leo 的 LoRA v1.2 适配、测试和调用脚本。

这套不是 FastH3/Sol-H3，也没有我方微调的新权重。首期交付侧重文生、
首帧图生和音频；不把 Ref2VA 包含在这个 LoRA 的支持承诺中。

已保留完整源码、上游来源、精确改动、模型下载版本和哈希、启动脚本、
API 示例以及 TT 适配注意事项。大模型权重不放进 Git，由脚本从固定的
公开版本下载。接收方可独立下载模型并部署。

两个必须保留的参数：v1.2 LoRA 的 **alpha=8、rank=128**；请求的
**num_inference_steps=5 对应四次去噪前向**，video/audio shift 分别为 6/3。

启动脚本会检查实际导入源码，避免误用镜像里未修改的旧版本。
整理仓库期间只做 CPU 验证，没有启动视频推理或占用 GPU；具体结果见
[验证说明](docs/VALIDATION.md)。同事明确说明 768p 未达到实时，此次交付
也不以实时性能作为验收条件。

TT 可直接从英文 [README](README.md) 和 [适配说明](docs/TT_PORTING.md) 开始。
仓库为私有，接收方需要 GitHub 读取权限。
