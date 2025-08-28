# LLM Fine-tuning Script

A simple Python script for fine-tuning Large Language Models using Hugging Face Transformers and LoRA.

## Features

- 🤖 **Easy to Use**: Simple scripts for quick fine-tuning
- 💾 **Memory Efficient**: Uses LoRA and 4-bit quantization
- 📊 **Multiple Data Formats**: JSON, CSV, or Hugging Face datasets
- 🎯 **Auto-Detection**: Automatically detects correct model settings

## Quick Start

### 1. Install Dependencies

```powershell
pip install -r requirements.txt
```

### 2. Run Quick Training

```powershell
python quick_train.py
```

This will:
- Create sample data automatically
- Fine-tune a small model
- Test the results

### 3. Try Examples

```powershell
python example_usage.py
```

## Data Format

Your training data should be a JSON file with input/output pairs:

```json
[
  {
    "input": "What is Python?",
    "output": "Python is a programming language..."
  },
  {
    "input": "How do I code?",
    "output": "Start with simple projects..."
  }
]
```

## Configuration

Edit `config.json` to customize your training:

```json
{
  "model_name": "microsoft/DialoGPT-small",
  "dataset_path": "data/my_data.json",
  "num_train_epochs": 3,
  "learning_rate": 2e-5
}
```

## Common Commands

```powershell
# Create sample data
python fine_tune_llm.py --create_sample

# Basic training
python fine_tune_llm.py --model_name gpt2 --dataset_path data/my_data.json

# Use config file
python fine_tune_llm.py --config config.json
```

## Troubleshooting

### Out of Memory?
- Use smaller models: `distilgpt2`, `microsoft/DialoGPT-small`
- Reduce batch size in config: `"per_device_train_batch_size": 1`
- Enable quantization: `"use_4bit": true`

### Model Not Working?
- The script auto-detects correct settings for most models
- Check that your data format matches the examples above
- Try the test scripts: `python validate_fix.py`

## Files Overview

- `quick_train.py` - Fastest way to start training
- `example_usage.py` - Examples for different scenarios
- `fine_tune_llm.py` - Main training script with all options
- `config.json` - Configuration template
- `requirements.txt` - Required packages

## Requirements

- Python 3.8+
- GPU with 4GB+ VRAM (recommended)
- 16GB+ System RAM

For CPU-only training, use smaller models and reduce batch sizes.
