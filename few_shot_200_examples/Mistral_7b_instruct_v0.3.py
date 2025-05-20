import os
import pandas as pd
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
import time
import argparse
from datetime import datetime
from huggingface_hub import login
import re
import gc

"""
Few-shot learning for regional bias detection using Mistral-7B with a full-context approach.
This implementation places all 200 examples in the prompt context to leverage the model's 
instruction-following capabilities for classification decisions.
"""

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Few-shot learning for regional bias detection with Mistral 7B')
    
    # Data paths
    parser.add_argument('--examples_path', type=str, required=True,
                        help='Path to CSV file with few-shot examples')
    parser.add_argument('--test_path', type=str, required=True,
                        help='Path to CSV file with test dataset')
    parser.add_argument('--output_dir', type=str, default='results',
                        help='Directory to save results')
    
    # Model configuration
    parser.add_argument('--model_name', type=str, 
                        default='mistralai/Mistral-7B-Instruct-v0.3',
                        help='Model name or path')
    parser.add_argument('--gpu_id', type=int, default=0,
                        help='GPU ID to use')
    parser.add_argument('--hf_token', type=str, default='',
                        help='HuggingFace token for model access')
    
    # Execution parameters
    parser.add_argument('--random_seed', type=int, default=42,
                        help='Random seed for reproducibility')
    parser.add_argument('--test_limit', type=int, default=None,
                        help='Limit number of test examples (for testing)')
    parser.add_argument('--cache_dir', type=str, default='model_cache',
                        help='Directory for model cache')
    
    return parser.parse_args()

def clean_text(text):
    """Clean and normalize text for model input"""
    text = str(text).lower()
    text = re.sub(r"http\S+", "", text)  # Remove URLs
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)  # Remove special characters
    text = re.sub(r"\s+", " ", text).strip()  # Remove extra spaces
    return text

def load_datasets(examples_path, test_path, test_limit=None):
    """
    Load the example and test datasets
    
    Args:
        examples_path: Path to examples CSV
        test_path: Path to test CSV
        test_limit: Optional limit for test dataset size
        
    Returns:
        examples_df, test_df (DataFrames)
    """
    # Load few-shot examples
    print(f"Loading examples from {examples_path}")
    examples_df = pd.read_csv(examples_path)
    
    # Load full dataset for testing
    print(f"Loading test dataset from {test_path}")
    test_df = pd.read_csv(test_path)
    
    print(f"Loaded {len(examples_df)} examples and {len(test_df)} test comments")
    
    # Verify class distribution in examples
    if 'Binary_Class' in examples_df.columns:
        bias_examples = examples_df[examples_df['Binary_Class'] == 1]
        non_bias_examples = examples_df[examples_df['Binary_Class'] == 0]
    else:
        bias_examples = examples_df[examples_df['Level-1'] >= 1]
        non_bias_examples = examples_df[examples_df['Level-1'] == 0]
    
    print(f"Found {len(bias_examples)} regional bias examples and {len(non_bias_examples)} non-regional bias examples")
    
    # Ensure there's no overlap between examples and test data
    examples_comments = set(examples_df['Comment'].str.strip())
    test_df = test_df[~test_df['Comment'].str.strip().isin(examples_comments)]
    print(f"After removing overlapping comments, {len(test_df)} test comments remain")
    
    # Clean comments
    print("Cleaning comment text...")
    test_df["Cleaned_Comment"] = test_df["Comment"].apply(clean_text)
    examples_df["Cleaned_Comment"] = examples_df["Comment"].apply(clean_text)
    
    # Apply test limit if specified
    if test_limit is not None and test_limit > 0:
        print(f"Limiting test set to {test_limit} examples")
        test_df = test_df.head(test_limit)
    
    return examples_df, test_df

def create_mistral_prompt(examples_df, comment, random_seed=42):
    """
    Create a prompt for few-shot learning with Mistral's instruction format.
    
    Args:
        examples_df: DataFrame with example comments
        comment: The comment to classify
        random_seed: Random seed for shuffling examples
        
    Returns:
        Formatted prompt string
    """
    # Check which column to use for classification
    class_column = 'Binary_Class' if 'Binary_Class' in examples_df.columns else 'Level-1'
    
    # Combine all examples and shuffle
    all_examples = examples_df.copy()
    all_examples = all_examples.sample(frac=1, random_state=random_seed)  # Shuffle all examples
    
    # Create Mistral-specific prompt with proper instruction format
    prompt = "[INST] You are an expert in identifying regional biases in comments about Indian states and regions. "
    prompt += "Task: Classify if the comment contains regional bias related to Indian states or regions.\n\n"
    prompt += "Instructions:\n"
    prompt += "- Regional Bias (1): Comments that contain stereotypes, prejudices, or biases about specific Indian states or regions.\n"
    prompt += "- Non-Regional Bias (0): Comments that don't contain regional stereotypes or biases about Indian states.\n\n"
    prompt += "Examples:\n"
    
    for i, row in all_examples.iterrows():
        # Get the classification value (0 or 1)
        if class_column == 'Binary_Class':
            classification = int(row['Binary_Class'])
        else:
            classification = 1 if row['Level-1'] >= 1 else 0
            
        prompt += f"Comment: \"{row['Cleaned_Comment']}\"\n"
        prompt += f"Classification: {classification}\n\n"
    
    prompt += f"Now classify this comment:\n\"{comment}\"\nClassification: [/INST]"
    
    return prompt

def setup_model(model_name, cache_dir, gpu_id, hf_token):
    """
    Load model and tokenizer with Mistral-specific settings
    
    Args:
        model_name: Model name or path 
        cache_dir: Directory for model cache
        gpu_id: GPU ID to use
        hf_token: HuggingFace token
        
    Returns:
        model, tokenizer, device
    """
    # Create cache directory
    os.makedirs(cache_dir, exist_ok=True)
    
    # Set environment variables for caching
    os.environ["TRANSFORMERS_CACHE"] = cache_dir
    
    # Set GPU device if specified
    if torch.cuda.is_available():
        os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
        device = torch.device("cuda:0")
        print(f"Using GPU {gpu_id}: {torch.cuda.get_device_name(0)}")
        print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    else:
        device = torch.device("cpu")
        print("CUDA not available. Using CPU.")
    
    # Clear cache before loading model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # Login to HuggingFace if token provided
    if hf_token:
        login(token=hf_token)
        print("Logged in to HuggingFace")
    
    print(f"Loading model: {model_name}")
    start_time = time.time()
    
    # Configure tokenizer - Mistral has specific tokenizer settings
    tokenizer_kwargs = {
        'model_max_length': 8192,  # Mistral supports longer contexts
    }
    
    if hf_token:
        tokenizer_kwargs['token'] = hf_token
    if cache_dir:
        tokenizer_kwargs['cache_dir'] = cache_dir
    
    tokenizer = AutoTokenizer.from_pretrained(model_name, **tokenizer_kwargs)
    
    # Ensure tokenizer has padding token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Configure quantization for 7B model
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16
    )
    
    # Load model - Mistral has specific loading requirements
    model_kwargs = {
        'quantization_config': quantization_config,
        'low_cpu_mem_usage': True,
    }
    
    if hf_token:
        model_kwargs['token'] = hf_token
    if cache_dir:
        model_kwargs['cache_dir'] = cache_dir
    
    if torch.cuda.is_available():
        model_kwargs['device_map'] = "auto"
    
    # Load the model
    model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
    
    # Set to evaluation mode
    model.eval()
    
    elapsed_time = time.time() - start_time
    print(f"Model loaded in {elapsed_time:.2f} seconds")
    
    return model, tokenizer, device

def predict_with_mistral(model, tokenizer, prompt, device, max_tokens=10):
    """
    Generate prediction using Mistral model with optimized parameters
    
    Args:
        model: The model
        tokenizer: The tokenizer for the model
        prompt: Input prompt text
        device: Device to use
        max_tokens: Maximum number of tokens to generate
        
    Returns:
        Predicted class (0 or 1), raw_output
    """
    # Tokenize the prompt with truncation to ensure it fits in context window
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=8192).to(device)
    
    # Generate response - Mistral needs specific generation parameters
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=0.1,  # Low temperature for more deterministic outputs
            do_sample=False,  # Don't sample for deterministic generation
            num_beams=2,      # Using beam search for better quality
            pad_token_id=tokenizer.pad_token_id
        )
    
    # Decode the generated text
    full_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract just the model's response (after our prompt)
    prompt_length = len(prompt)
    if full_output.startswith(prompt):
        generated_text = full_output[prompt_length:].strip()
    else:
        generated_text = full_output[-50:].strip()  # Just get the end
    
    # Clear tensors to prevent OOM
    del inputs, outputs
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # Look for explicit "1" or "0" at the end or in the last few characters
    if generated_text.endswith("1") or generated_text == "1":
        return 1, full_output
    elif generated_text.endswith("0") or generated_text == "0":
        return 0, full_output
    
    # If we don't have a clear number, look more thoroughly at the last 50 chars
    last_part = full_output[-50:].lower()
    
    if "1" in last_part and not "0" in last_part:
        return 1, full_output
    elif "0" in last_part and not "1" in last_part:
        return 0, full_output
    elif "regional bias" in last_part or "bias" in last_part:
        return 1, full_output
    elif "non-regional" in last_part or "no bias" in last_part:
        return 0, full_output
    
    # Default to non-regional bias if we can't determine
    return 0, full_output

def batch_predict(model, tokenizer, test_df, examples_df, device, random_seed=42):
    """
    Process comments in batches for inference with Mistral-specific optimizations
    
    Args:
        model: The model
        tokenizer: The tokenizer
        test_df: DataFrame with test data
        examples_df: DataFrame with examples
        device: Device to use
        random_seed: Random seed for reproducibility
        
    Returns:
        predictions, raw_outputs
    """
    predictions = []
    raw_outputs = []
    
    # Get test comments
    test_comments = test_df["Cleaned_Comment"].tolist()
    
    # Process examples individually - Mistral needs more aggressive memory handling
    for i in range(0, len(test_comments)):
        comment = test_comments[i]
        
        # Generate prompt with all examples
        prompt = create_mistral_prompt(examples_df, comment, random_seed)
        
        # Tokenize to check length
        tokens = tokenizer(prompt, return_tensors="pt", truncation=False)
        input_ids_length = tokens.input_ids.shape[1]
        
        if input_ids_length > 8000:  # If too long, use a simplified prompt
            print(f"Prompt is very long ({input_ids_length} tokens). Using truncation.")
            # Simplified prompt for long inputs
            prompt = "[INST] You are an expert in identifying regional biases in comments about Indian states and regions. "
            prompt += "Task: Classify if the comment contains regional bias related to Indian states or regions.\n\n"
            prompt += "Instructions:\n"
            prompt += "- Regional Bias (1): Comments that contain stereotypes, prejudices, or biases about specific Indian states or regions.\n"
            prompt += "- Non-Regional Bias (0): Comments that don't contain regional stereotypes or biases about Indian states.\n\n"
            prompt += f"Classify this comment:\n\"{comment}\"\nClassification: [/INST]"
        
        # Clear token tensors
        del tokens
        
        # Add a brief pause before prediction for better quality with Mistral
        if i > 0 and i % 5 == 0:
            print("Pausing briefly to optimize model performance...")
            time.sleep(1)  # 1-second pause every 5 examples
        
        # Get prediction
        prediction, raw_output = predict_with_mistral(model, tokenizer, prompt, device)
        
        # Store results
        predictions.append(prediction)
        raw_outputs.append(raw_output)
        
        # Log progress
        if (i + 1) % 10 == 0 or i == 0:
            print(f"Processed example {i+1}/{len(test_comments)}")
            print(f"Decision: {prediction} (0=non-regional, 1=regional)")
        
        # Clear cache to prevent OOM errors - Mistral needs more frequent clearing
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()
    
    return predictions, raw_outputs

def save_results(test_df, predictions, raw_outputs, output_dir, model_name="mistral_7b_200examples"):
    """
    Save prediction results and evaluation metrics
    
    Args:
        test_df: DataFrame with test data
        predictions: List of predictions
        raw_outputs: List of raw model outputs
        output_dir: Directory to save results
        model_name: Name of model for file naming
        
    Returns:
        accuracy, f1_score
    """
    # Create output directories
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "visualizations"), exist_ok=True)
    
    # Get true labels
    true_labels = test_df['Level-1'].apply(lambda x: 1 if x >= 1 else 0).tolist()
    
    # Save predictions with raw outputs
    results_df = test_df.copy()
    results_df['Predicted'] = predictions
    
    # Truncate raw outputs to prevent huge files
    truncated_outputs = [str(output)[:500] for output in raw_outputs]
    results_df['Model_Output'] = truncated_outputs
    
    # Create output paths
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    predictions_path = os.path.join(output_dir, f"{model_name}_predictions_{timestamp}.csv")
    report_path = os.path.join(output_dir, f"{model_name}_report_{timestamp}.txt")
    matrix_path = os.path.join(output_dir, "visualizations", f"{model_name}_confusion_matrix_{timestamp}.png")
    summary_path = os.path.join(output_dir, "visualizations", f"{model_name}_results_summary_{timestamp}.png")
    
    # Save predictions CSV
    results_df.to_csv(predictions_path, index=False)
    print(f"Predictions saved to {predictions_path}")
    
    # Generate classification report
    report = classification_report(true_labels, predictions)
    with open(report_path, 'w') as f:
        f.write(f"Classification Report for {model_name}\n\n")
        f.write(f"Timestamp: {timestamp}\n\n")
        f.write(report)
    print(f"Classification report saved to {report_path}")
    
    # Calculate metrics
    accuracy = accuracy_score(true_labels, predictions)
    f1 = f1_score(true_labels, predictions)
    
    # Log metrics
    print(f"Accuracy: {accuracy:.4f}")
    print(f"F1 Score: {f1:.4f}")
    
    # Generate confusion matrix
    cm = confusion_matrix(true_labels, predictions)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Non-Regional Bias', 'Regional Bias'],
                yticklabels=['Non-Regional Bias', 'Regional Bias'])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(f'Confusion Matrix - {model_name}')
    plt.tight_layout()
    plt.savefig(matrix_path)
    print(f"Confusion matrix saved to {matrix_path}")
    
    # Create a comprehensive results visualization
    plt.figure(figsize=(12, 8))
    
    # Create a 2x2 grid of subplots
    plt.subplot(2, 2, 1)
    # Plot confusion matrix as heatmap
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Non-Regional', 'Regional'],
                yticklabels=['Non-Regional', 'Regional'])
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    
    # Plot class distribution
    plt.subplot(2, 2, 2)
    class_counts = pd.Series(true_labels).value_counts().sort_index()
    plt.bar(['Non-Regional', 'Regional'], [class_counts.get(0, 0), class_counts.get(1, 0)], 
            color=['#1f77b4', '#ff7f0e'])
    plt.title('Test Set Class Distribution')
    plt.ylabel('Number of Samples')
    
    # Add value labels on bars
    plt.text(0, class_counts.get(0, 0) + 5, str(class_counts.get(0, 0)), ha='center')
    plt.text(1, class_counts.get(1, 0) + 5, str(class_counts.get(1, 0)), ha='center')
    
    # Plot prediction distribution
    plt.subplot(2, 2, 3)
    pred_counts = pd.Series(predictions).value_counts().sort_index()
    plt.bar(['Non-Regional', 'Regional'], [pred_counts.get(0, 0), pred_counts.get(1, 0)], 
            color=['#2ca02c', '#d62728'])
    plt.title('Model Predictions')
    plt.ylabel('Number of Samples')
    
    # Add value labels on bars
    plt.text(0, pred_counts.get(0, 0) + 5, str(pred_counts.get(0, 0)), ha='center')
    plt.text(1, pred_counts.get(1, 0) + 5, str(pred_counts.get(1, 0)), ha='center')
    
    # Plot accuracy and F1
    plt.subplot(2, 2, 4)
    plt.bar(['Accuracy', 'F1 Score'], [accuracy, f1], color=['#9467bd', '#8c564b'])
    plt.ylim(0, 1.0)
    plt.title('Model Performance')
    plt.text(0, accuracy + 0.05, f"{accuracy:.4f}", ha='center')
    plt.text(1, f1 + 0.05, f"{f1:.4f}", ha='center')
    
    plt.tight_layout()
    plt.savefig(summary_path)
    print(f"Results summary visualization saved to {summary_path}")
    
    return accuracy, f1

def main():
    """Main execution function"""
    # Parse arguments
    args = parse_arguments()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.cache_dir, exist_ok=True)
    
    # Set random seed for reproducibility
    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    print(f"Random seed set to {args.random_seed}")
    
    # Start timing
    start_time = time.time()
    
    # Load datasets
    examples_df, test_df = load_datasets(
        args.examples_path, args.test_path, args.test_limit
    )
    
    # Set up model and tokenizer
    model, tokenizer, device = setup_model(
        args.model_name, args.cache_dir, args.gpu_id, args.hf_token
    )
    
    # Predict using our examples dataset with all 200 examples
    print(f"Processing {len(test_df)} comments with all {len(examples_df)} few-shot examples...")
    
    predictions, raw_outputs = batch_predict(
        model, tokenizer, test_df, examples_df, device,
        random_seed=args.random_seed
    )
    
    # Save results
    accuracy, f1 = save_results(
        test_df, predictions, raw_outputs, args.output_dir
    )
    
    # End timing
    end_time = time.time()
    elapsed_hours = (end_time - start_time) / 3600
    print(f"Total execution time: {elapsed_hours:.2f} hours")
    
    # Final summary
    print("===== Final Summary =====")
    print(f"Model: {args.model_name}")
    print(f"Test set size: {len(test_df)}")
    print(f"Few-shot examples: {len(examples_df)}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"F1 Score: {f1:.4f}")

if __name__ == "__main__":
    main()
