# JetEngine

A lightweight Inference Engine built for [SDAR](https://huggingface.co/collections/JetLM/sdar-689b1b6d392a4eeb2664f8ff) series based on nano-vllm.

## Installation

### Environment Setup

```
transformers>=4.52.4
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