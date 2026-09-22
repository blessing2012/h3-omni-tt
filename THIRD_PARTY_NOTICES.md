# Third-party sources and model terms

- vLLM-Omni source: https://github.com/vllm-project/vllm-omni ; Apache-2.0.
  The original license and file-level attribution are retained under
  `vendor/vllm-omni/`. Subcomponents retain their own applicable notices/licenses.
- MiniMax-H3 weights: https://huggingface.co/MiniMaxAI/MiniMax-H3 ; MiniMax H3
  Community License Agreement, as published with that model. Base weights are
  downloaded separately; this repository does not relicense them.
- LightX2V Turbo adapter: https://huggingface.co/lightx2v/Minimax-h3-Turbo ; the
  upstream model card declares Apache-2.0. Base-model terms remain separate.
- Reference container: `vllm/vllm-omni:v0.28.0`, pinned by digest in
  `provenance.json`; its bundled dependencies retain their respective licenses.

The root integration documentation and scripts are a private LinearGameAI partner
handoff. No blanket open-source license is added for those new materials; upstream
rights and licenses are preserved. Model weights, credentials, proprietary datasets
and production infrastructure are not bundled.
