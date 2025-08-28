"""
Quick training script for fine-tuning LLMs
This is a simplified version for immediate use
"""

import torch
from fine_tune_llm import LLMFineTuner, FineTuningConfig, create_sample_dataset
import os


def quick_train():
    """Quick training function with optimized settings"""
    
    print("🚀 Quick LLM Fine-tuning Script")
    print("=" * 40)
    
    # Check if CUDA is available
    if torch.cuda.is_available():
        print(f"✅ GPU detected: {torch.cuda.get_device_name(0)}")
        print(f"   GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    else:
        print("⚠️  No GPU detected. Training will be slow on CPU.")
    
    # Create sample dataset if it doesn't exist
    dataset_path = "data/sample_data.json"
    if not os.path.exists(dataset_path):
        print("📁 Creating sample dataset...")
        create_sample_dataset(dataset_path)
    
    # Optimized configuration for quick training
    config = FineTuningConfig(
        model_name="microsoft/DialoGPT-small",  # Small model for quick training
        dataset_path=dataset_path,
        output_dir="models/quick_fine_tuned",
        
        # Training settings optimized for speed and memory
        num_train_epochs=2,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=2,
        learning_rate=5e-5,
        max_length=256,  # Shorter sequences for speed
        
        # Enable memory optimizations
        use_lora=True,
        lora_r=8,  # Smaller rank for speed
        use_4bit=True,  # 4-bit quantization
        fp16=True,
        
        # Faster evaluation
        eval_steps=25,
        save_steps=25,
        logging_steps=5,
        
        # Early stopping
        early_stopping_patience=2,
        load_best_model_at_end=True
    )
    
    print("\n⚙️  Configuration:")
    print(f"   Model: {config.model_name}")
    print(f"   Epochs: {config.num_train_epochs}")
    print(f"   Batch Size: {config.per_device_train_batch_size}")
    print(f"   Learning Rate: {config.learning_rate}")
    print(f"   LoRA: {'Enabled' if config.use_lora else 'Disabled'}")
    print(f"   4-bit Quantization: {'Enabled' if config.use_4bit else 'Disabled'}")
    
    # Initialize fine-tuner
    fine_tuner = LLMFineTuner(config)
    
    try:
        print("\n🔄 Loading model and tokenizer...")
        fine_tuner.load_tokenizer_and_model()
        
        print("📊 Loading and preprocessing datasets...")
        fine_tuner.load_datasets()
        fine_tuner.prepare_datasets()
        
        print("⚡ Setting up training...")
        fine_tuner.setup_training()
        
        print("\n🏋️  Starting training...")
        print("=" * 40)
        train_result, eval_metrics = fine_tuner.train()
        
        print("\n✅ Training completed!")
        print("=" * 40)
        print(f"📈 Final evaluation loss: {eval_metrics.get('eval_loss', 'N/A'):.4f}")
        print(f"💾 Model saved to: {config.output_dir}")
        
        # Quick test of the fine-tuned model
        print("\n🧪 Testing the fine-tuned model...")
        test_prompt = "What is machine learning?"
        print(f"Prompt: {test_prompt}")
        
        responses = fine_tuner.generate_text(
            test_prompt, 
            max_length=100, 
            num_return_sequences=1
        )
        print(f"Response: {responses[0]}")
        
        print("\n🎉 Fine-tuning completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Error during fine-tuning: {str(e)}")
        print("\n💡 Troubleshooting tips:")
        print("   - Ensure you have enough GPU memory")
        print("   - Try reducing batch_size or max_length")
        print("   - Install missing dependencies from requirements.txt")
        print("   - Use a smaller model like 'distilgpt2'")


if __name__ == "__main__":
    quick_train()
