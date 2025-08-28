"""
Example script demonstrating how to use the LLM fine-tuning script
"""

import json
import os
from fine_tune_llm import LLMFineTuner, FineTuningConfig, create_sample_dataset


def create_custom_dataset():
    """Create a custom dataset for question-answering fine-tuning"""
    
    # Sample data for a simple Q&A chatbot
    qa_data = [
        {
            "input": "What is Python?",
            "output": "Python is a high-level, interpreted programming language known for its simplicity and readability. It's widely used for web development, data analysis, artificial intelligence, and automation."
        },
        {
            "input": "How do I install Python packages?",
            "output": "You can install Python packages using pip. For example: 'pip install package_name'. You can also use conda if you're using Anaconda: 'conda install package_name'."
        },
        {
            "input": "What is machine learning?",
            "output": "Machine learning is a subset of artificial intelligence that enables computers to learn and make decisions from data without being explicitly programmed for every scenario."
        },
        {
            "input": "Explain neural networks",
            "output": "Neural networks are computational models inspired by the human brain. They consist of interconnected nodes (neurons) that process information through weighted connections and activation functions."
        },
        {
            "input": "What is deep learning?",
            "output": "Deep learning is a subset of machine learning that uses neural networks with multiple layers (deep networks) to learn complex patterns and representations from large amounts of data."
        },
        {
            "input": "How do I start learning programming?",
            "output": "Start with a beginner-friendly language like Python. Practice with simple projects, take online courses, read documentation, and build small applications to reinforce your learning."
        },
        {
            "input": "What is the difference between supervised and unsupervised learning?",
            "output": "Supervised learning uses labeled data to train models for prediction tasks, while unsupervised learning finds patterns in unlabeled data without predefined target outputs."
        },
        {
            "input": "What are the main steps in data preprocessing?",
            "output": "Data preprocessing typically involves data cleaning, handling missing values, feature scaling, encoding categorical variables, and splitting data into training and testing sets."
        }
    ]
    
    # Create data directory if it doesn't exist
    os.makedirs("data", exist_ok=True)
    
    # Save the dataset
    with open("data/qa_dataset.json", "w", encoding="utf-8") as f:
        json.dump(qa_data, f, indent=2, ensure_ascii=False)
    
    print("Custom Q&A dataset created at: data/qa_dataset.json")


def run_basic_fine_tuning():
    """Run basic fine-tuning with default settings"""
    
    print("=== Basic Fine-tuning Example ===")
    
    # Create sample dataset
    create_custom_dataset()
    
    # Configuration for basic fine-tuning
    config = FineTuningConfig(
        model_name="microsoft/DialoGPT-small",  # Using smaller model for demo
        dataset_path="data/qa_dataset.json",
        output_dir="models/qa_chatbot",
        num_train_epochs=2,  # Reduced for demo
        per_device_train_batch_size=2,
        learning_rate=5e-5,
        use_lora=True,
        use_4bit=True,  # Enable 4-bit quantization to save memory
        max_length=256,  # Reduced for demo
        save_steps=50,
        eval_steps=50,
        logging_steps=10
    )
    
    # Initialize fine-tuner
    fine_tuner = LLMFineTuner(config)
    
    try:
        # Run fine-tuning pipeline
        fine_tuner.load_tokenizer_and_model()
        fine_tuner.load_datasets()
        fine_tuner.prepare_datasets()
        fine_tuner.setup_training()
        
        print("Starting training...")
        train_result, eval_metrics = fine_tuner.train()
        
        print("Fine-tuning completed!")
        print(f"Final evaluation loss: {eval_metrics.get('eval_loss', 'N/A')}")
        
        # Test the fine-tuned model
        print("\n=== Testing Fine-tuned Model ===")
        test_prompts = [
            "What is Python?",
            "How do I learn programming?",
            "Explain machine learning"
        ]
        
        for prompt in test_prompts:
            print(f"\nPrompt: {prompt}")
            responses = fine_tuner.generate_text(prompt, max_length=150)
            print(f"Response: {responses[0]}")
            
    except Exception as e:
        print(f"Error during fine-tuning: {str(e)}")
        print("This might be due to insufficient GPU memory or missing dependencies.")
        print("Try reducing batch_size or using a smaller model.")


def run_advanced_fine_tuning():
    """Run advanced fine-tuning with custom configuration"""
    
    print("=== Advanced Fine-tuning Example ===")
    
    # Advanced configuration
    config = FineTuningConfig(
        model_name="microsoft/DialoGPT-medium",
        dataset_path="data/qa_dataset.json",
        output_dir="models/advanced_chatbot",
        num_train_epochs=3,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=2,
        learning_rate=2e-5,
        weight_decay=0.01,
        warmup_ratio=0.1,
        use_lora=True,
        lora_r=32,  # Higher rank for better adaptation
        lora_alpha=64,
        use_4bit=True,
        max_length=512,
        early_stopping_patience=2,
        use_wandb=False,  # Set to True if you want to use Weights & Biases
        fp16=True
    )
    
    fine_tuner = LLMFineTuner(config)
    
    try:
        fine_tuner.load_tokenizer_and_model()
        fine_tuner.load_datasets()
        fine_tuner.prepare_datasets()
        fine_tuner.setup_training()
        
        train_result, eval_metrics = fine_tuner.train()
        print("Advanced fine-tuning completed!")
        
    except Exception as e:
        print(f"Error: {str(e)}")


def main():
    """Main function to run examples"""
    
    print("LLM Fine-tuning Examples")
    print("========================")
    
    choice = input("\nChoose an example:\n1. Basic Fine-tuning\n2. Advanced Fine-tuning\n3. Create sample dataset only\nEnter choice (1-3): ")
    
    if choice == "1":
        run_basic_fine_tuning()
    elif choice == "2":
        create_custom_dataset()  # Ensure dataset exists
        run_advanced_fine_tuning()
    elif choice == "3":
        create_custom_dataset()
        print("Sample dataset created! You can now use it with the fine-tuning script.")
    else:
        print("Invalid choice. Please run again and select 1, 2, or 3.")


if __name__ == "__main__":
    main()
