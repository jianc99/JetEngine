# JetEngine

JetEngine, a lightweight inference engine for the SDAR series built on [nano-vllm](https://github.com/GeeeekExplorer/nano-vllm) support both dense and MoE models and Tensor Parallel distributed inference, delivers tons of acceleration compared to the naive implementation.

In our benchmark, we tested the 4B SDAR model with block size 4 (basic acceleration setting) and batch size 128:
- On NVIDIA A800, JetEngine reached 1800+ tokens/second.
- On NVIDIA H200, JetEngine achieved 3700+ tokens/second using FlashAttention-2 + Triton kernels.

This demonstrates that JetEngine can unlock production-level throughput for SDAR models, making it ideal for both research-scale batch inference and real-world deployment scenarios.
## Installation

### Environment Setup

```
transformers>=4.52.4
flash-attn
```

```bash
pip install flash-attn --no-build-isolation
git clone https://github.com/Labman42/JetEngine.git
cd JetEngine
pip install .
```

## Manual Download
If you prefer to download the model weights manually, use the following command:
```bash
huggingface-cli download --resume-download JetLM/SDAR-1.7B-Chat \
  --local-dir ~/huggingface/SDAR-1.7B-Chat/ \
  --local-dir-use-symlinks False
```

## Quick Start

```bash
python example.py
```

See `example.py` for usage. The API mirrors vLLM's interface with some differences in the `LLM.generate` method.

## 📬 Contact

For issues or inquiries:
- **Yihan Bian**, University of Maryland, College Park (ybian@umd.edu)
