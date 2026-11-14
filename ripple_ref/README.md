---
license: apache-2.0
---
# Ripple-65B Preview NVFP4 (Experimental)

<div align="center">
<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://framerusercontent.com/images/rSaJGoPP1Saln5kpScZNGCw.png" width="30%">
  <source media="(prefers-color-scheme: light)" srcset="https://framerusercontent.com/images/3rFANSmoZuSBhjRwxZolyEU5guo.png" width="30%">
  <img alt="Ricursive Logo" src="https://framerusercontent.com/images/3rFANSmoZuSBhjRwxZolyEU5guo.png" width="30%">
</picture>
</div>

<p align="center">
  <a href="https://huggingface.co/ricursiveAI/Ripple-65B-Preview">BF16</a> |
  <a href="https://huggingface.co/ricursiveAI/Ripple-65B-Preview-NVFP4">NVFP4</a> |
  <a href="https://huggingface.co/ricursiveAI/Ripple-65B-Base-Preview">Base Model</a> |
  <a href="https://huggingface.co/ricursiveAI/Ripple-65B-Preview-DFlash2">DFlash2</a> |
  <a href="https://www.ricursive.com/ripple">Launch Blog</a> | 
  License: <a href="https://huggingface.co/ricursiveAI/Ripple-65B-Preview/blob/main/LICENSE">Apache 2.0</a>
</p>

Ricursive Intelligence introduces **Ripple-65B Preview**, a 65B parameter dense Transformer model pretrained on 12 trillion tokens with support for up to 1M context length. 
Ripple-65B Preview delivers strong results on agentic coding, scoring **80.52 on TerminalBench 2.1** and
surpassing significantly-larger US open-weight models such as Reflection Beam 501B (**80.1**), Meta Muse Spark Open 816B (**76.4**), 
Thinking Machines Inkling 975B (**63.8**), and NVIDIA Nemotron 3 Ultra 550B (**56.4**). 

Our Preview release includes an instruction-tuned model in BF16 and NVFP4 numerics, and a BF16 base model. **This repo contains the NVFP4 instruction-tuned checkpoint.**

This is a conservative quantization of the BF16 checkpoint, using NVFP4 for the MLP layers while keeping attention in BF16.

## Citation

```bibtex
@misc{ricursive2026ripple65b,
      title={Ripple-65B: Pushing the Frontier of Local Intelligence},
      author={Ricursive Intelligence},
      year={2026},
}
```

## Contact

For questions and comments, please contact us at [info@ricursive.com](mailto:info@ricursive.com). To learn more, please visit our [website](https://www.ricursive.com).