#!/usr/bin/env python3
"""
Few-shot classification for regional bias detection using DeepSeek-R1-Distill-Qwen-1.5B
with k-nearest examples selection approach.

Author: Debasmita
"""

import os
import sys
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
import logging
import argparse
from datetime import datetime
import re
import gc
from huggingface_hub import login

# Default paths
DEFAULT_EXAMPLES_PATH = '/DATA2/akash/venvs/debasmita/data/balanced_dataset_200_comments.csv'
DEFAULT_ANNOTATED_PATH = '/DATA2/akash/venvs/debasmita/data/annotated_experiment_phase_new - Sheet1.csv'
DEFAULT_OUTPUT_DIR = '/DATA2/akash/venvs/debasmita/results/few_shot_k'
DEFAULT_CACHE_DIR = '/DATA2/akash/venvs/debasmita/model_cache'
DEFAULT_MODEL_NAME = 'deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B'
DEFAULT_HF_TOKEN = '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'
DEFAULT_K_EXAMPLES = 20
DEFAULT_GPU_ID = 1

def parse_arguments():
    """A simpler approach to parse arguments with defaults"""
    # Create a new parser
    parser = argparse.ArgumentParser(description='Few-shot learning for regional bias detection')
    
    # Add all arguments with their proper defaults
    parser.add_argument('--examples_path', type=str, default=DEFAULT_EXAMPLES_PATH,
                        help='Path to CSV file with few-shot examples')
    parser.add_argument('--annotated_path', type=str, default=DEFAULT_ANNOTATED_PATH,
                        help='Path to CSV file with annotated dataset')
    parser.add_argument('--output_dir', type=str, default=DEFAULT_OUTPUT_DIR,
                        help='Directory to save results')
    parser.add_argument('--cache_dir', type=str, default=DEFAULT_CACHE_DIR,
                        help='Directory for model cache')
    parser.add_argument('--model_name', type=str, default=DEFAULT_MODEL_NAME,
                        help='Model name or path')
    parser.add_argument('--k_examples', type=int, default=DEFAULT_K_EXAMPLES,
                        help='Number of examples to use for few-shot learning')
    parser.add_argument('--gpu_id', type=int, default=DEFAULT_GPU_ID,
                        help='GPU ID to use')
    parser.add_argument('--hf_token', type=str, default=DEFAULT_HF_TOKEN,
                        help='HuggingFace token')
    parser.add_argument('--test_limit', type=int, default=None,
                        help='Limit number of test examples')
    parser.add_argument('--save_checkpoints', action='store_true',
                        help='Save checkpoints during processing')
    parser.add_argument('--checkpoint_interval', type=int, default=50,
                        help='Interval for saving checkpoints')
    
    return parser.parse_args()

def setup_logging(output_dir):
    """Set up logging configuration"""
    os.makedirs(output_dir, exist_ok=True)
    
    log_file = os.path.join(output_dir, f"few_shot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    return logging.getLogger(__name__)

def clean_text(text):
    """Clean and normalize text for model input"""
    text = str(text).lower()
    text = re.sub(r"http\S+", "", text)  # Remove URLs
    text = re.sub(r"\s+", " ", text).strip()  # Remove extra spaces
    return text

def load_datasets(examples_path, annotated_path, logger):
    """
    Load examples and test datasets, ensuring no overlap
    
    Args:
        examples_path: Path to CSV with few-shot examples
        annotated_path: Path to CSV with annotated dataset
        logger: Logger instance
        
    Returns:
        examples_df, test_df (DataFrames)
    """
    # Load few-shot examples
    logger.info(f"Loading examples from {examples_path}")
    examples_df = pd.read_csv(examples_path)
    
    # Load full dataset for testing
    logger.info(f"Loading annotated dataset from {annotated_path}")
    annotated_df = pd.read_csv(annotated_path)
    
    # Process the datasets
    logger.info(f"Loaded {len(examples_df)} examples from examples dataset")
    logger.info(f"Loaded {len(annotated_df)} total comments from annotated dataset")
    
    # Clean examples data
    examples_df["Cleaned_Comment"] = examples_df["Comment"].apply(clean_text)
    
    # Make sure we have binary labels for examples
    if 'Binary_Class' in examples_df.columns:
        # Already have binary class
        examples_df['Label'] = examples_df['Binary_Class']
    elif 'Level-1' in examples_df.columns:
        # Convert Level-1 to binary
        examples_df['Label'] = examples_df['Level-1'].apply(lambda x: 1 if x >= 1 else 0)
    else:
        # Try to infer from Score column
        if 'Score' in examples_df.columns:
            examples_df['Label'] = examples_df['Score'].apply(lambda x: 1 if x > 0 else 0)
        else:
            logger.error("Could not find label column in examples dataset. Expected 'Binary_Class', 'Level-1', or 'Score'.")
            raise ValueError("No label column found in examples dataset")
    
    # Report class balance in examples
    pos_examples = len(examples_df[examples_df['Label'] == 1])
    neg_examples = len(examples_df[examples_df['Label'] == 0])
    logger.info(f"Examples set has {pos_examples} regional bias and {neg_examples} non-regional bias examples")
    
    # Clean annotated data
    annotated_df["Cleaned_Comment"] = annotated_df["Comment"].apply(clean_text)
    
    # Make sure we have binary labels for annotated data
    if 'Binary_Class' in annotated_df.columns:
        annotated_df['Label'] = annotated_df['Binary_Class']
    elif 'Level-1' in annotated_df.columns:
        annotated_df['Label'] = annotated_df['Level-1'].apply(lambda x: 1 if x >= 1 else 0)
    else:
        if 'Score' in annotated_df.columns:
            annotated_df['Label'] = annotated_df['Score'].apply(lambda x: 1 if x > 0 else 0)
        else:
            logger.error("Could not find label column in annotated dataset")
            raise ValueError("No label column found in annotated dataset")
    
    # Remove overlap between examples and test data
    examples_comments = set(examples_df['Cleaned_Comment'].str.strip())
    test_df = annotated_df[~annotated_df['Cleaned_Comment'].str.strip().isin(examples_comments)]
    
    overlap_count = len(annotated_df) - len(test_df)
    logger.info(f"Removed {overlap_count} overlapping comments between examples and test data")
    logger.info(f"Final test set size: {len(test_df)} comments")
    
    return examples_df, test_df

def setup_model(model_name, cache_dir, gpu_id, hf_token, logger):
    """
    Set up model and tokenizer
    
    Args:
        model_name: Model name or path
        cache_dir: Directory for model cache
        gpu_id: GPU ID to use
        hf_token: HuggingFace token
        logger: Logger instance
        
    Returns:
        model, tokenizer, device
    """
    # Set GPU device if specified
    if torch.cuda.is_available():
        os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
        device = torch.device("cuda:0")  # After setting CUDA_VISIBLE_DEVICES, the selected GPU becomes device 0
        logger.info(f"Using GPU {gpu_id}: {torch.cuda.get_device_name(0)}")
        logger.info(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    else:
        device = torch.device("cpu")
        logger.info("CUDA not available. Using CPU.")
    
    # Set up cache directory
    os.makedirs(cache_dir, exist_ok=True)
    os.environ['TRANSFORMERS_CACHE'] = cache_dir
    os.environ['HF_HOME'] = cache_dir
    logger.info(f"Using cache directory: {cache_dir}")
    
    # Clear GPU cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # Login to HuggingFace if token provided
    if hf_token:
        login(token=hf_token)
        logger.info("Logged in to HuggingFace")
    
    logger.info(f"Loading model: {model_name}")
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
    
    try:
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
        logger.info(f"Model loaded in {elapsed_time:.2f} seconds")
        
        return model, tokenizer, device
        
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        raise

def create_few_shot_prompt(examples_df, comment, k=20):
    """
    Create a prompt with k-nearest examples
    
    Args:
        examples_df: DataFrame with examples
        comment: The comment to classify
        k: Number of examples to use (default 20)
        
    Returns:
        Formatted prompt string
    """
    # Use a subset of most relevant examples
    # Convert to list for processing
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

def predict_with_model(model, tokenizer, prompt, device, max_tokens=10, logger=None):
    """
    Generate prediction using model
    
    Args:
        model: The model
        tokenizer: The tokenizer
        prompt: Input prompt text
        device: Device to use (cuda or cpu)
        max_tokens: Maximum number of tokens to generate
        logger: Logger instance for detailed logging
        
    Returns:
        predicted_class (0 or 1), raw_output
    """
    try:
        # Tokenize the prompt
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=8192)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Log token count if logger provided
        if logger and logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"Tokenized prompt length: {inputs['input_ids'].shape[1]} tokens")
        
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
        
        # Log full output if logger provided
        if logger and logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"Generated text: '{generated_text}'")
        
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
            
        # If we can't find a clear answer in first few chars, examine content
        if "regional bias" in generated_text.lower():
            return 1, generated_text
        elif "non-regional" in generated_text.lower() or "no bias" in generated_text.lower():
            return 0, generated_text
        
        # Default to 0 (non-regional bias) if unclear
        if logger:
            logger.warning(f"Could not extract classification from: '{generated_text}'")
        return 0, generated_text
        
    except Exception as e:
        if logger:
            logger.error(f"Error in prediction: {e}")
        return 0, f"ERROR: {str(e)}"

def batch_predict(model, tokenizer, test_df, examples_df, device, k_examples=20, 
                 save_checkpoints=False, checkpoint_interval=50, output_dir=None, logger=None):
    """
    Process test comments in batches
    
    Args:
        model: The model
        tokenizer: The tokenizer
        test_df: DataFrame with test data
        examples_df: DataFrame with examples
        device: Device to use
        k_examples: Number of examples to use per prediction
        save_checkpoints: Whether to save checkpoints
        checkpoint_interval: Interval for saving checkpoints
        output_dir: Directory to save checkpoints
        logger: Logger instance
        
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
        
        try:
            # Generate prompt with k nearest examples
            prompt = create_few_shot_prompt(examples_df, comment, k=k_examples)
            
            # Tokenize to check length
            tokens = tokenizer(prompt, return_tensors="pt", truncation=False)
            input_ids_length = tokens.input_ids.shape[1]
            
            if input_ids_length > 6000:  # If too long, reduce number of examples
                logger.warning(f"Prompt is very long ({input_ids_length} tokens). Reducing to {k_examples//2} examples.")
                prompt = create_few_shot_prompt(examples_df, comment, k=k_examples//2)
                
                # Check again to be safe
                tokens = tokenizer(prompt, return_tensors="pt", truncation=False)
                input_ids_length = tokens.input_ids.shape[1]
                
                if input_ids_length > 7500:  # If still too long, use minimal examples
                    logger.warning(f"Prompt still too long ({input_ids_length} tokens). Using only 5 examples.")
                    prompt = create_few_shot_prompt(examples_df, comment, k=5)
            
            # Clear token tensors
            del tokens
            
            # Get prediction
            prediction, raw_output = predict_with_model(model, tokenizer, prompt, device, logger=logger)
            
            # Store results
            predictions.append(prediction)
            raw_outputs.append(raw_output)
            
            # Log progress
            if (i + 1) % 10 == 0 or i == 0:
                logger.info(f"Processed example {i+1}/{len(test_comments)}")
                logger.info(f"Decision: {prediction} (0=non-regional, 1=regional)")
            
            # Save checkpoint if enabled
            if save_checkpoints and output_dir and ((i + 1) % checkpoint_interval == 0 or i == len(test_comments) - 1):
                checkpoint_df = pd.DataFrame({
                    'Comment': test_df['Comment'].iloc[:i+1].tolist(),
                    'Cleaned_Comment': test_df['Cleaned_Comment'].iloc[:i+1].tolist(),
                    'True_Label': test_df['Label'].iloc[:i+1].tolist(),
                    'Predicted': predictions[:i+1],
                    'Model_Output': raw_outputs[:i+1]
                })
                
                # Use model short name for file naming
                model_short_name = "deepseek-k-nearest"
                checkpoint_path = os.path.join(output_dir, f"{model_short_name}_checkpoint_{i+1}.csv")
                checkpoint_df.to_csv(checkpoint_path, index=False)
                logger.info(f"Saved checkpoint at {checkpoint_path}")
        
        except Exception as e:
            logger.error(f"Error processing comment {i+1}: {e}")
            predictions.append(0)  # Default to non-regional bias
            raw_outputs.append(f"ERROR: {str(e)}")
        
        # Clear cache to prevent OOM errors
        if torch.cuda.is_available() and (i + 1) % 5 == 0:
            torch.cuda.empty_cache()
            gc.collect()
    
    return predictions, raw_outputs

def save_results(test_df, predictions, raw_outputs, output_dir, logger, model_name="deepseek-k-nearest"):
    """
    Save prediction results and evaluation metrics
    
    Args:
        test_df: DataFrame with test data
        predictions: List of predictions
        raw_outputs: List of raw model outputs
        output_dir: Directory to save results
        logger: Logger instance
        model_name: Name of model for file naming
        
    Returns:
        accuracy, f1_score
    """
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
    matrix_path = os.path.join(output_dir, f"{model_name}_confusion_matrix_{timestamp}.png")
    summary_path = os.path.join(output_dir, f"{model_name}_results_summary_{timestamp}.png")
    
    # Save predictions CSV
    results_df.to_csv(predictions_path, index=False)
    logger.info(f"Predictions saved to {predictions_path}")
    
    # Generate classification report
    report = classification_report(true_labels, predictions)
    with open(report_path, 'w') as f:
        f.write(f"Classification Report for {model_name}\n\n")
        f.write(f"Timestamp: {timestamp}\n\n")
        f.write(report)
    logger.info(f"Classification report saved to {report_path}")
    
    # Calculate metrics
    accuracy = accuracy_score(true_labels, predictions)
    f1 = f1_score(true_labels, predictions)
    
    # Log metrics
    logger.info(f"Accuracy: {accuracy:.4f}")
    logger.info(f"F1 Score: {f1:.4f}")
    
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
    logger.info(f"Confusion matrix saved to {matrix_path}")
    
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
    logger.info(f"Results summary visualization saved to {summary_path}")
    
    return accuracy, f1

def main():
    """Main execution function"""
    # Parse arguments
    args = parse_arguments()
    
    # Set up logging
    logger = setup_logging(args.output_dir)
    
    # Log arguments
    logger.info("Arguments:")
    for arg, value in vars(args).items():
        logger.info(f"  {arg}: {value}")
    
    # Start timing
    start_time = time.time()
    
    try:
        # Load datasets
        examples_df, test_df = load_datasets(args.examples_path, args.annotated_path, logger)
        
        # Apply test limit if specified
        if args.test_limit is not None and args.test_limit > 0:
            logger.info(f"Limiting test set to {args.test_limit} examples")
            test_df = test_df.head(args.test_limit)
        
        # Set up model and tokenizer
        model, tokenizer, device = setup_model(
            args.model_name, args.cache_dir, args.gpu_id, args.hf_token, logger
        )
        
        # Predict using our examples dataset
        logger.info(f"Processing {len(test_df)} comments with {args.k_examples} few-shot examples...")
        
        # Extract model short name for file naming
        model_short_name = f"deepseek-k{args.k_examples}"
        
        predictions, raw_outputs = batch_predict(
            model, tokenizer, test_df, examples_df, device,
            k_examples=args.k_examples,
            save_checkpoints=args.save_checkpoints,
            checkpoint_interval=args.checkpoint_interval,
            output_dir=args.output_dir,
            logger=logger
        )
        
        # Save results
        accuracy, f1 = save_results(
            test_df, predictions, raw_outputs, args.output_dir, logger, 
            model_name=model_short_name
        )
        
        # End timing
        end_time = time.time()
        elapsed_minutes = (end_time - start_time) / 60
        logger.info(f"Total execution time: {elapsed_minutes:.2f} minutes")
        
        # Final summary
        logger.info("===== Final Summary =====")
        logger.info(f"Model: {args.model_name}")
        logger.info(f"Test set size: {len(test_df)}")
        logger.info(f"Few-shot examples: {len(examples_df)}")
        logger.info(f"K-nearest examples used: {args.k_examples}")
        logger.info(f"Accuracy: {accuracy:.4f}")
        logger.info(f"F1 Score: {f1:.4f}")
    
    except Exception as e:
        logger.error(f"Error in main execution: {e}", exc_info=True)
        raise

if __name__ == "__main__":
    main()