import os
import pandas as pd
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import seaborn as sns
import time
import argparse
from datetime import datetime
import re
import gc
from huggingface_hub import login

"""
Few-shot learning for regional bias detection using TF-IDF nearest neighbor example selection.
This implementation uses k nearest examples to create context for LLM prediction.
"""

def parse_arguments():
    """Parse command line arguments with sensible defaults"""
    parser = argparse.ArgumentParser(description='Few-shot learning for regional bias detection')
    
    # Data paths
    parser.add_argument('--examples_path', type=str, required=True,
                        help='Path to CSV file with few-shot examples')
    parser.add_argument('--annotated_path', type=str, required=True,
                        help='Path to CSV file with annotated dataset')
    parser.add_argument('--output_dir', type=str, default='results',
                        help='Directory to save results')
    
    # Model configuration
    parser.add_argument('--model_name', type=str, 
                        default='deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B',
                        help='Model name or path')
    parser.add_argument('--k_examples', type=int, default=20,
                        help='Number of examples to use for few-shot learning')
    parser.add_argument('--gpu_id', type=int, default=0,
                        help='GPU ID to use')
    parser.add_argument('--hf_token', type=str, default='',
                        help='HuggingFace token (recommended to use env var instead)')
    
    # Additional parameters
    parser.add_argument('--test_limit', type=int, default=None,
                        help='Limit number of test examples')
    parser.add_argument('--cache_dir', type=str, default='model_cache',
                        help='Directory for model cache')
    
    return parser.parse_args()

def clean_text(text):
    """Clean and normalize text for model input"""
    text = str(text).lower()
    text = re.sub(r"http\S+", "", text)  # Remove URLs
    text = re.sub(r"\s+", " ", text).strip()  # Remove extra spaces
    return text

def load_datasets(examples_path, annotated_path):
    """
    Load examples and test datasets, ensuring no overlap
    
    Args:
        examples_path: Path to CSV with few-shot examples
        annotated_path: Path to CSV with annotated dataset
        
    Returns:
        examples_df, test_df (DataFrames)
    """
    # Load few-shot examples
    print(f"Loading examples from {examples_path}")
    examples_df = pd.read_csv(examples_path)
    
    # Load full dataset for testing
    print(f"Loading annotated dataset from {annotated_path}")
    annotated_df = pd.read_csv(annotated_path)
    
    print(f"Loaded {len(examples_df)} examples and {len(annotated_df)} comments")
    
    # Clean examples data
    examples_df["Cleaned_Comment"] = examples_df["Comment"].apply(clean_text)
    
    # Make sure we have binary labels for examples
    if 'Binary_Class' in examples_df.columns:
        examples_df['Label'] = examples_df['Binary_Class']
    elif 'Level-1' in examples_df.columns:
        examples_df['Label'] = examples_df['Level-1'].apply(lambda x: 1 if x >= 1 else 0)
    elif 'Score' in examples_df.columns:
        examples_df['Label'] = examples_df['Score'].apply(lambda x: 1 if x > 0 else 0)
    
    # Report class balance in examples
    pos_examples = len(examples_df[examples_df['Label'] == 1])
    neg_examples = len(examples_df[examples_df['Label'] == 0])
    print(f"Examples set has {pos_examples} regional bias and {neg_examples} non-regional bias examples")
    
    # Clean annotated data
    annotated_df["Cleaned_Comment"] = annotated_df["Comment"].apply(clean_text)
    
    # Make sure we have binary labels for annotated data
    if 'Binary_Class' in annotated_df.columns:
        annotated_df['Label'] = annotated_df['Binary_Class']
    elif 'Level-1' in annotated_df.columns:
        annotated_df['Label'] = annotated_df['Level-1'].apply(lambda x: 1 if x >= 1 else 0)
    elif 'Score' in annotated_df.columns:
        annotated_df['Label'] = annotated_df['Score'].apply(lambda x: 1 if x > 0 else 0)
    
    # Remove overlap between examples and test data
    examples_comments = set(examples_df['Cleaned_Comment'].str.strip())
    test_df = annotated_df[~annotated_df['Cleaned_Comment'].str.strip().isin(examples_comments)]
    
    overlap_count = len(annotated_df) - len(test_df)
    print(f"Removed {overlap_count} overlapping comments between examples and test data")
    print(f"Final test set size: {len(test_df)} comments")
    
    return examples_df, test_df

def setup_model(model_name, cache_dir, gpu_id, hf_token):
    """
    Set up model and tokenizer
    
    Args:
        model_name: Model name or path
        cache_dir: Directory for model cache
        gpu_id: GPU ID to use
        hf_token: HuggingFace token
        
    Returns:
        model, tokenizer, device
    """
    # Create cache directory if it doesn't exist
    os.makedirs(cache_dir, exist_ok=True)
    
    # Set GPU device if specified
    if torch.cuda.is_available():
        os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
        device = torch.device("cuda:0")
        print(f"Using GPU {gpu_id}: {torch.cuda.get_device_name(0)}")
        print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    else:
        device = torch.device("cpu")
        print("CUDA not available. Using CPU.")
    
    # Set up cache directory
    os.environ['TRANSFORMERS_CACHE'] = cache_dir
    
    # Clear GPU cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # Login to HuggingFace if token provided
    if hf_token:
        login(token=hf_token)
        print("Logged in to HuggingFace")
    
    print(f"Loading model: {model_name}")
    start_time = time.time()
    
    # Setup tokenizer
    tokenizer_kwargs = {
        'trust_remote_code': True,
        'model_max_length': 8192,
    }
    
    if hf_token:
        tokenizer_kwargs['token'] = hf_token
    if cache_dir:
        tokenizer_kwargs['cache_dir'] = cache_dir
    
    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, **tokenizer_kwargs)
    
    # Ensure tokenizer has padding token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Configure quantization for efficient memory usage
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16
    )
    
    # Load model
    model_kwargs = {
        'quantization_config': quantization_config,
        'low_cpu_mem_usage': True,
        'trust_remote_code': True,
    }
    
    if hf_token:
        model_kwargs['token'] = hf_token
    if cache_dir:
        model_kwargs['cache_dir'] = cache_dir
    
    if torch.cuda.is_available():
        model_kwargs['device_map'] = "auto"
    
    # Load the model with optimized parameters
    model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
    
    # Set to evaluation mode
    model.eval()
    
    elapsed_time = time.time() - start_time
    print(f"Model loaded in {elapsed_time:.2f} seconds")
    
    return model, tokenizer, device

def create_few_shot_prompt(examples_df, comment, k=20):
    """
    Create a prompt with k-nearest examples based on TF-IDF similarity
    
    Args:
        examples_df: DataFrame with examples
        comment: The comment to classify
        k: Number of examples to use (default 20)
        
    Returns:
        Formatted prompt string
    """
    # Use a subset of most relevant examples
    examples_texts = examples_df['Cleaned_Comment'].tolist()
    examples_labels = examples_df['Label'].tolist()
    
    # Create TF-IDF vectors
    vectorizer = TfidfVectorizer(max_features=5000)
    all_texts = examples_texts + [comment]  # Include the comment for transformation
    tfidf_matrix = vectorizer.fit_transform(all_texts)
    
    # Get comment vector (last item)
    comment_vector = tfidf_matrix[-1]
    examples_vectors = tfidf_matrix[:-1]  # All except last
    
    # Calculate similarities
    similarities = cosine_similarity(comment_vector, examples_vectors).flatten()
    
    # Get k most similar examples of each class
    bias_indices = [i for i, label in enumerate(examples_labels) if label == 1]
    non_bias_indices = [i for i, label in enumerate(examples_labels) if label == 0]
    
    bias_similarities = [(i, similarities[i]) for i in bias_indices]
    non_bias_similarities = [(i, similarities[i]) for i in non_bias_indices]
    
    # Sort by similarity
    bias_similarities.sort(key=lambda x: x[1], reverse=True)
    non_bias_similarities.sort(key=lambda x: x[1], reverse=True)
    
    # Get top k/2 from each class
    k_per_class = k // 2
    top_bias = bias_similarities[:k_per_class]
    top_non_bias = non_bias_similarities[:k_per_class]
    
    # Combine and sort for better interspersing
    selected_indices = [i for i, _ in top_bias] + [i for i, _ in top_non_bias]
    
    # Create prompt with selected examples only
    prompt = "You are an expert in identifying regional biases in comments about Indian states and regions. "
    prompt += "Task: Classify if the comment contains regional bias related to Indian states or regions.\n\n"
    prompt += "Instructions:\n"
    prompt += "- Regional Bias (1): Comments that contain stereotypes, prejudices, or biases about specific Indian states or regions.\n"
    prompt += "- Non-Regional Bias (0): Comments that don't contain regional stereotypes or biases about Indian states.\n\n"
    prompt += "Examples:\n"
    
    # Add selected examples
    for idx in selected_indices:
        example_text = examples_texts[idx]
        example_label = examples_labels[idx]
        prompt += f"Comment: \"{example_text}\"\n"
        prompt += f"Classification: {example_label}\n\n"
    
    prompt += f"Now classify this comment:\n\"{comment}\"\nClassification:"
    
    return prompt

def predict_with_model(model, tokenizer, prompt, device, max_tokens=10):
    """
    Generate prediction using model
    
    Args:
        model: The model
        tokenizer: The tokenizer
        prompt: Input prompt text
        device: Device to use (cuda or cpu)
        max_tokens: Maximum number of tokens to generate
        
    Returns:
        predicted_class (0 or 1), raw_output
    """
    # Tokenize the prompt
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=8192)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # Generate response
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=0.1,
            do_sample=False,
            num_beams=1,
            pad_token_id=tokenizer.pad_token_id
        )
    
    # Get just the new tokens (the model's answer)
    generated_ids = outputs[0][inputs['input_ids'].shape[1]:]
    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
    
    # Clear tensors to prevent OOM
    del inputs, outputs, generated_ids
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # Extract classification from generated text
    if generated_text.startswith("1") or generated_text == "1":
        return 1, generated_text
    elif generated_text.startswith("0") or generated_text == "0":
        return 0, generated_text
    
    # If not found directly, examine the first few characters
    if generated_text and generated_text[0] in ["0", "1"]:
        return int(generated_text[0]), generated_text
        
    # If still not found, look through the text
    if "1" in generated_text[:5] and not "0" in generated_text[:5]:
        return 1, generated_text
    elif "0" in generated_text[:5] and not "1" in generated_text[:5]:
        return 0, generated_text
        
    # If we can't find a clear answer, examine content
    if "regional bias" in generated_text.lower():
        return 1, generated_text
    elif "non-regional" in generated_text.lower() or "no bias" in generated_text.lower():
        return 0, generated_text
    
    # Default to 0 (non-regional bias) if unclear
    return 0, generated_text

def batch_predict(model, tokenizer, test_df, examples_df, device, k_examples=20):
    """
    Process test comments and make predictions
    
    Args:
        model: The model
        tokenizer: The tokenizer
        test_df: DataFrame with test data
        examples_df: DataFrame with examples
        device: Device to use
        k_examples: Number of examples to use per prediction
        
    Returns:
        predictions, raw_outputs
    """
    predictions = []
    raw_outputs = []
    
    # Get test comments
    test_comments = test_df["Cleaned_Comment"].tolist()
    
    # Process each comment
    for i in range(0, len(test_comments)):
        comment = test_comments[i]
        
        # Generate prompt with k nearest examples
        prompt = create_few_shot_prompt(examples_df, comment, k=k_examples)
        
        # Tokenize to check length
        tokens = tokenizer(prompt, return_tensors="pt", truncation=False)
        input_ids_length = tokens.input_ids.shape[1]
        
        if input_ids_length > 6000:  # If too long, reduce number of examples
            print(f"Prompt is very long ({input_ids_length} tokens). Reducing to {k_examples//2} examples.")
            prompt = create_few_shot_prompt(examples_df, comment, k=k_examples//2)
            
            # Check again to be safe
            tokens = tokenizer(prompt, return_tensors="pt", truncation=False)
            input_ids_length = tokens.input_ids.shape[1]
            
            if input_ids_length > 7500:  # If still too long, use minimal examples
                print(f"Prompt still too long ({input_ids_length} tokens). Using only 5 examples.")
                prompt = create_few_shot_prompt(examples_df, comment, k=5)
        
        # Clear token tensors
        del tokens
        
        # Get prediction
        prediction, raw_output = predict_with_model(model, tokenizer, prompt, device)
        
        # Store results
        predictions.append(prediction)
        raw_outputs.append(raw_output)
        
        # Log progress
        if (i + 1) % 10 == 0 or i == 0:
            print(f"Processed example {i+1}/{len(test_comments)}")
            print(f"Decision: {prediction} (0=non-regional, 1=regional)")
        
        # Clear cache to prevent OOM errors
        if torch.cuda.is_available() and (i + 1) % 5 == 0:
            torch.cuda.empty_cache()
            gc.collect()
    
    return predictions, raw_outputs

def save_results(test_df, predictions, raw_outputs, output_dir, model_name="deepseek-k-nearest"):
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
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "visualizations"), exist_ok=True)
    
    # Get true labels
    true_labels = test_df['Label'].tolist()
    
    # Save predictions with raw outputs
    results_df = test_df.copy()
    results_df['Predicted'] = predictions
    results_df['Model_Output'] = raw_outputs
    
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
    class_counts = results_df['Label'].value_counts().sort_index()
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
    
    # Start timing
    start_time = time.time()
    
    # Load datasets
    examples_df, test_df = load_datasets(args.examples_path, args.annotated_path)
    
    # Apply test limit if specified
    if args.test_limit is not None and args.test_limit > 0:
        print(f"Limiting test set to {args.test_limit} examples")
        test_df = test_df.head(args.test_limit)
    
    # Set up model and tokenizer
    model, tokenizer, device = setup_model(
        args.model_name, args.cache_dir, args.gpu_id, args.hf_token
    )
    
    # Predict using our examples dataset
    print(f"Processing {len(test_df)} comments with {args.k_examples} few-shot examples...")
    
    # Extract model short name for file naming
    model_short_name = f"deepseek-k{args.k_examples}"
    
    # Make predictions
    predictions, raw_outputs = batch_predict(
        model, tokenizer, test_df, examples_df, device,
        k_examples=args.k_examples
    )
    
    # Save results
    accuracy, f1 = save_results(
        test_df, predictions, raw_outputs, args.output_dir, 
        model_name=model_short_name
    )
    
    # End timing
    end_time = time.time()
    elapsed_minutes = (end_time - start_time) / 60
    print(f"Total execution time: {elapsed_minutes:.2f} minutes")
    
    # Final summary
    print("===== Final Summary =====")
    print(f"Model: {args.model_name}")
    print(f"Test set size: {len(test_df)}")
    print(f"Few-shot examples: {len(examples_df)}")
    print(f"K-nearest examples used: {args.k_examples}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"F1 Score: {f1:.4f}")

if __name__ == "__main__":
    main()
