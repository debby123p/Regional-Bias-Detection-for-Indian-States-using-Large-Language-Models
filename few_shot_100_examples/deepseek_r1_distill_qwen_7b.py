import os
import sys
import pandas as pd
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import time
import logging
import argparse
from datetime import datetime
from huggingface_hub import login
import re

# Default model configuration parameters
DEFAULT_MODEL_NAME = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
DEFAULT_GPU_ID = 0
DEFAULT_RANDOM_SEED = 42
DEFAULT_MAX_LENGTH = 4096
DEFAULT_MAX_TOKENS = 50

def parse_arguments():
    """Parse command line arguments for regional bias detection experiment"""
    parser = argparse.ArgumentParser(description='Few-shot learning for regional bias detection with DeepSeek (100 examples)')
    
    parser.add_argument('--examples_path', type=str, 
                        default=os.environ.get('EXAMPLES_PATH', 'data/50_examples_few_shot_classification_dataset.csv'),
                        help='Path to CSV file with 100 few-shot examples (50 per class)')
    
    parser.add_argument('--test_path', type=str, 
                        default=os.environ.get('TEST_PATH', 'data/annotated_dataset.csv'),
                        help='Path to CSV file with test dataset')
    
    parser.add_argument('--output_dir', type=str, 
                        default=os.environ.get('OUTPUT_DIR', 'results/deepseek_few_shot_100'),
                        help='Directory to save results')
    
    parser.add_argument('--cache_dir', type=str, 
                        default=os.environ.get('CACHE_DIR', 'model_cache'),
                        help='Directory for model cache')
    
    parser.add_argument('--log_dir', type=str, 
                        default=os.environ.get('LOG_DIR', 'logs'),
                        help='Directory for log files')
    
    parser.add_argument('--model_name', type=str, 
                        default=os.environ.get('MODEL_NAME', DEFAULT_MODEL_NAME),
                        help='Model name or path')
    
    parser.add_argument('--gpu_id', type=int, 
                        default=int(os.environ.get('GPU_ID', DEFAULT_GPU_ID)),
                        help='GPU ID to use')
    
    parser.add_argument('--hf_token', type=str, 
                        default=os.environ.get('HF_TOKEN', ''),
                        help='HuggingFace token (recommended to use env var instead)')
    
    parser.add_argument('--random_seed', type=int,
                        default=int(os.environ.get('RANDOM_SEED', DEFAULT_RANDOM_SEED)),
                        help='Random seed for reproducibility')
    
    parser.add_argument('--test_limit', type=int, 
                        default=None,
                        help='Limit number of test examples (for testing)')
    
    parser.add_argument('--checkpoint_interval', type=int, 
                        default=10,
                        help='Interval for saving checkpoints')
    
    parser.add_argument('--max_length', type=int, 
                        default=int(os.environ.get('MAX_LENGTH', DEFAULT_MAX_LENGTH)),
                        help='Maximum context length for tokenization')
    
    parser.add_argument('--max_tokens', type=int, 
                        default=int(os.environ.get('MAX_TOKENS', DEFAULT_MAX_TOKENS)),
                        help='Maximum number of tokens to generate')
    
    return parser.parse_args()

def setup_logging(log_dir, model_name):
    """Set up logging configuration for tracking experimental results"""
    # Create log directory
    os.makedirs(log_dir, exist_ok=True)
    
    # Configure log file with timestamp
    log_file = os.path.join(log_dir, f"{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    
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
    """Clean and normalize text for consistent model input"""
    text = str(text).lower()
    text = re.sub(r"http\S+", "", text)  # Remove URLs
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)  # Remove special characters
    text = re.sub(r"\s+", " ", text).strip()  # Remove extra spaces
    return text

def load_datasets(examples_path, test_path, logger, test_limit=None):
    """
    Load and prepare the datasets for few-shot learning
    
    Args:
        examples_path: Path to examples CSV
        test_path: Path to test CSV
        logger: Logger instance
        test_limit: Optional limit for test dataset size
        
    Returns:
        examples_df, test_df (DataFrames)
    """
    # Load few-shot examples 
    logger.info(f"Loading examples from {examples_path}")
    examples_df = pd.read_csv(examples_path)
    
    # Load test dataset
    logger.info(f"Loading test dataset from {test_path}")
    test_df = pd.read_csv(test_path)
    
    logger.info(f"Loaded {len(examples_df)} examples and {len(test_df)} test comments")
    
    # Split the examples into bias and non-bias
    bias_examples = examples_df[examples_df['Level-1'] >= 1]
    non_bias_examples = examples_df[examples_df['Level-1'] == 0]
    logger.info(f"Found {len(bias_examples)} bias examples and {len(non_bias_examples)} non-bias examples")
    
    # Ensure there's no overlap between examples and test data
    examples_comments = set(examples_df['Comment'].str.strip())
    test_df = test_df[~test_df['Comment'].str.strip().isin(examples_comments)]
    logger.info(f"After removing overlapping comments, {len(test_df)} test comments remain")
    
    # Clean comments for better model processing
    logger.info("Cleaning comment text...")
    test_df["Cleaned_Comment"] = test_df["Comment"].apply(clean_text)
    examples_df["Cleaned_Comment"] = examples_df["Comment"].apply(clean_text)
    
    # Apply test limit if specified
    if test_limit is not None and test_limit > 0:
        logger.info(f"Limiting test set to {test_limit} examples")
        test_df = test_df.head(test_limit)
    
    return examples_df, test_df

def create_few_shot_prompt(examples_df, comment, random_seed=42):
    """
    Create a prompt for few-shot learning with examples and the target comment
    
    Args:
        examples_df: DataFrame with example comments
        comment: The comment to classify
        random_seed: Random seed for shuffling examples
        
    Returns:
        Formatted prompt string optimized for LLM reasoning
    """
    # Combine all examples and shuffle
    all_examples = examples_df.copy()
    all_examples = all_examples.sample(frac=1, random_state=random_seed)  # Shuffle examples
    
    # Create the prompt - optimized for DeepSeek-R1 reasoning capabilities
    prompt = "You are an expert in identifying regional biases in comments about Indian states and regions. "
    prompt += "Task: Classify if the comment contains regional bias related to Indian states or regions.\n\n"
    prompt += "Instructions:\n"
    prompt += "- Regional Bias (1): Comments that contain stereotypes, prejudices, or biases about specific Indian states or regions.\n"
    prompt += "- Non-Regional Bias (0): Comments that don't contain regional stereotypes or biases about Indian states.\n\n"
    prompt += "Think step-by-step before making your final decision.\n\n"
    prompt += "Examples:\n"
    
    # Add examples to the prompt
    for i, row in all_examples.iterrows():
        # Convert Level-1 to binary classification (0 or 1)
        classification = 1 if row['Level-1'] >= 1 else 0
        
        prompt += f"Comment: \"{row['Cleaned_Comment']}\"\n"
        prompt += f"Classification: {classification}\n\n"
    
    # Add the current comment to classify with detailed reasoning instructions
    prompt += f"Now carefully classify this comment:\n\"{comment}\"\n"
    prompt += "Let me think this through carefully:\n"
    prompt += "1. Is this comment about an Indian state, region, or people from a specific region?\n"
    prompt += "2. Does the comment contain stereotypes, generalizations, or prejudice?\n"
    prompt += "3. Is the content biased or discriminatory?\n\n"
    prompt += "Based on the above analysis, the classification is:"
    
    return prompt

def setup_model(model_name, cache_dir, gpu_id, hf_token, logger):
    """
    Load and configure the model and tokenizer
    
    Args:
        model_name: Model name or path 
        cache_dir: Directory for model cache
        gpu_id: GPU ID to use
        hf_token: HuggingFace token
        logger: Logger instance
        
    Returns:
        model, tokenizer, device
    """
    # Ensure cache directory exists
    os.makedirs(cache_dir, exist_ok=True)
    
    # Set environment variables for model caching
    os.environ["TRANSFORMERS_CACHE"] = cache_dir
    os.environ["HF_HOME"] = cache_dir
    os.environ["HF_DATASETS_CACHE"] = cache_dir
    
    # Configure GPU device
    if torch.cuda.is_available():
        os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
        torch.cuda.set_device(0)
        device = torch.device("cuda:0")
        logger.info(f"Using GPU {gpu_id}: {torch.cuda.get_device_name(0)}")
        logger.info(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        
        # Clear GPU cache
        torch.cuda.empty_cache()
    else:
        device = torch.device("cpu")
        logger.info("CUDA not available. Using CPU.")
    
    # Login to HuggingFace if token provided
    if hf_token:
        login(token=hf_token)
        logger.info("Logged in to HuggingFace")
    
    logger.info(f"Loading model: {model_name}")
    start_time = time.time()
    
    # Configure tokenizer settings
    tokenizer_kwargs = {}
    if hf_token:
        tokenizer_kwargs['token'] = hf_token
    if cache_dir:
        tokenizer_kwargs['cache_dir'] = cache_dir
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, **tokenizer_kwargs)
    
    # Ensure tokenizer has padding token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Configure 8-bit quantization for memory efficiency
    quantization_config = BitsAndBytesConfig(
        load_in_8bit=True,
        bnb_8bit_use_double_quant=True,
        bnb_8bit_compute_dtype=torch.float16
    )
    
    # Configure model loading parameters
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
    logger.info(f"Model loaded in {elapsed_time:.2f} seconds")
    
    return model, tokenizer, device

def predict_with_model(model, tokenizer, prompt, device, max_length=4096, max_tokens=50, logger=None):
    """
    Generate prediction using model with classification extraction
    
    Args:
        model: The model
        tokenizer: The tokenizer for the model
        prompt: Input prompt text
        device: Device to use
        max_length: Maximum context length for tokenization
        max_tokens: Maximum number of tokens to generate
        logger: Logger instance for detailed logging
        
    Returns:
        Predicted class (0 or 1), raw_output
    """
    # Tokenize the prompt with truncation to ensure it fits in context window
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_length).to(device)
    
    # Generate response with optimal parameters for reasoning
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=0.1,      # Low temperature for deterministic outputs
            do_sample=False,      # Don't sample for deterministic generation
            num_beams=3,          # Beam search for better reasoning
            pad_token_id=tokenizer.eos_token_id
        )
    
    # Decode the generated text
    full_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract the model's response (after our prompt)
    prompt_length = len(prompt)
    if full_output.startswith(prompt):
        generated_text = full_output[prompt_length:].strip()
    else:
        # If we can't find the exact prompt (tokenization differences)
        generated_text = full_output[-100:].strip()
    
    # Clear tensors to prevent OOM
    del inputs, outputs
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # Classification extraction strategy
    # First look for explicit "1" or "0"
    if "1" in generated_text.split() or generated_text.endswith("1"):
        return 1, full_output
    elif "0" in generated_text.split() or generated_text.endswith("0"):
        return 0, full_output
    
    # If no explicit number, look for bias keywords
    if any(term in generated_text.lower() for term in ["regional bias", "stereotype", "prejudice", "discriminat", "bias"]):
        return 1, full_output
    elif any(term in generated_text.lower() for term in ["non-regional", "not regional", "no bias", "neutral"]):
        return 0, full_output
    
    # More advanced classification based on keyword frequency
    bias_indicators = ["generaliz", "stereo", "prejud", "discriminat", "bias"]
    non_bias_indicators = ["neutral", "factual", "objective", "no bias", "not bias"]
    
    bias_score = sum(1 for term in bias_indicators if term in generated_text.lower())
    non_bias_score = sum(1 for term in non_bias_indicators if term in generated_text.lower())
    
    if bias_score > non_bias_score:
        return 1, full_output
    else:
        return 0, full_output

def batch_predict(model, tokenizer, test_df, examples_df, device, random_seed=42, max_length=4096, 
                 max_tokens=50, checkpoint_interval=10, output_dir=None, logger=None):
    """
    Process test comments in batch with token length management
    
    Args:
        model: The model
        tokenizer: The tokenizer
        test_df: DataFrame with test data
        examples_df: DataFrame with examples
        device: Device to use
        random_seed: Random seed for reproducibility
        max_length: Maximum context length
        max_tokens: Maximum tokens to generate
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
    
    # Create checkpoint directory if needed
    if output_dir:
        checkpoint_dir = os.path.join(output_dir, "checkpoints")
        os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Process examples one by one
    for i, comment in enumerate(test_comments):
        # Generate prompt with all examples
        prompt = create_few_shot_prompt(examples_df, comment, random_seed)
        
        # Check tokenized length and use simplified prompt if too long
        tokens = tokenizer(prompt, return_tensors="pt", truncation=False)
        input_ids_length = tokens.input_ids.shape[1]
        
        if input_ids_length > max_length - 100:  # Leave room for generation
            logger.warning(f"Prompt is very long ({input_ids_length} tokens). Using simplified prompt.")
            # Create simplified prompt for long contexts
            prompt = "You are an expert in identifying regional biases in comments about Indian states and regions. "
            prompt += "Task: Classify if the comment contains regional bias related to Indian states or regions.\n\n"
            prompt += "Instructions:\n"
            prompt += "- Regional Bias (1): Comments that contain stereotypes, prejudices, or biases about specific Indian states or regions.\n"
            prompt += "- Non-Regional Bias (0): Comments that don't contain regional stereotypes or biases about Indian states.\n\n"
            prompt += f"Classify this comment:\n\"{comment}\"\n"
            prompt += "Think carefully and respond with only 0 or 1:"
        
        # Clear token tensors
        del tokens
        
        # Get prediction
        prediction, raw_output = predict_with_model(model, tokenizer, prompt, device, max_length, max_tokens, logger)
        
        # Store results
        predictions.append(prediction)
        raw_outputs.append(raw_output)
        
        # Log progress at intervals
        if (i + 1) % 10 == 0 or i == 0:
            logger.info(f"Processed example {i+1}/{len(test_comments)}")
            logger.info(f"Decision: {prediction} (0=non-regional, 1=regional)")
        
        # Save checkpoint if enabled
        if output_dir and ((i + 1) % checkpoint_interval == 0 or i == len(test_comments) - 1):
            # Create a short model name for file naming
            model_short_name = os.path.basename(model_name).replace('/', '_')
            
            checkpoint_df = pd.DataFrame({
                'Comment': test_df['Comment'].iloc[:i+1].tolist(),
                'Cleaned_Comment': test_df['Cleaned_Comment'].iloc[:i+1].tolist(),
                'True_Label': test_df['Level-1'].iloc[:i+1].apply(lambda x: 1 if x >= 1 else 0).tolist(),
                'Predicted': predictions[:i+1],
                'Model_Output': [str(output)[:500] for output in raw_outputs[:i+1]]  # Truncate long outputs
            })
            checkpoint_path = os.path.join(output_dir, "checkpoints", f"{model_short_name}_checkpoint_{i+1}.csv")
            checkpoint_df.to_csv(checkpoint_path, index=False)
            logger.info(f"Saved checkpoint at {checkpoint_path}")
        
        # Clear cache to prevent OOM errors
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    return predictions, raw_outputs

def save_results(test_df, predictions, raw_outputs, output_dir, logger, model_name):
    """
    Save prediction results and generate evaluation metrics
    
    Args:
        test_df: DataFrame with test data
        predictions: List of predictions
        raw_outputs: List of raw model outputs
        output_dir: Directory to save results
        logger: Logger instance
        model_name: Name of model for file naming
        
    Returns:
        accuracy
    """
    # Get true labels
    true_labels = test_df['Level-1'].apply(lambda x: 1 if x >= 1 else 0).tolist()
    
    # Create directories
    os.makedirs(output_dir, exist_ok=True)
    viz_dir = os.path.join(output_dir, "visualizations")
    os.makedirs(viz_dir, exist_ok=True)
    
    # Create results dataframe
    results_df = test_df.copy()
    results_df['Predicted'] = predictions
    results_df['Model_Output'] = [str(output)[:500] for output in raw_outputs]  # Truncate long outputs
    
    # Create output file paths with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_short_name = os.path.basename(model_name).replace('/', '_')
    predictions_path = os.path.join(output_dir, f"{model_short_name}_predictions_{timestamp}.csv")
    report_path = os.path.join(output_dir, f"{model_short_name}_report_{timestamp}.txt")
    matrix_path = os.path.join(viz_dir, f"{model_short_name}_confusion_matrix_{timestamp}.png")
    summary_path = os.path.join(viz_dir, f"{model_short_name}_results_summary_{timestamp}.png")
    
    # Save predictions CSV
    results_df.to_csv(predictions_path, index=False)
    logger.info(f"Predictions saved to {predictions_path}")
    
    # Generate classification report
    report = classification_report(true_labels, predictions)
    with open(report_path, 'w') as f:
        f.write(f"Classification Report for {model_short_name}\n\n")
        f.write(f"Timestamp: {timestamp}\n\n")
        f.write(report)
    logger.info(f"Classification report saved to {report_path}")
    
    # Calculate metrics
    accuracy = accuracy_score(true_labels, predictions)
    logger.info(f"Accuracy: {accuracy:.4f}")
    
    # Generate confusion matrix visualization
    cm = confusion_matrix(true_labels, predictions)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Non-Regional Bias', 'Regional Bias'],
                yticklabels=['Non-Regional Bias', 'Regional Bias'])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(f'Confusion Matrix - {model_short_name}')
    plt.tight_layout()
    plt.savefig(matrix_path)
    logger.info(f"Confusion matrix saved to {matrix_path}")
    
    # Create comprehensive results visualization
    plt.figure(figsize=(12, 8))
    
    # Plot confusion matrix
    plt.subplot(2, 2, 1)
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
    for i, v in enumerate([class_counts.get(0, 0), class_counts.get(1, 0)]):
        plt.text(i, v + 5, str(v), ha='center')
    
    # Plot prediction distribution
    plt.subplot(2, 2, 3)
    pred_counts = pd.Series(predictions).value_counts().sort_index()
    plt.bar(['Non-Regional', 'Regional'], [pred_counts.get(0, 0), pred_counts.get(1, 0)], 
            color=['#2ca02c', '#d62728'])
    plt.title('Model Predictions')
    plt.ylabel('Number of Samples')
    
    # Add value labels on bars
    for i, v in enumerate([pred_counts.get(0, 0), pred_counts.get(1, 0)]):
        plt.text(i, v + 5, str(v), ha='center')
    
    # Plot accuracy
    plt.subplot(2, 2, 4)
    plt.bar(['Accuracy'], [accuracy], color='#9467bd')
    plt.ylim(0, 1.0)
    plt.title('Model Accuracy')
    plt.text(0, accuracy + 0.05, f"{accuracy:.4f}", ha='center')
    
    plt.tight_layout()
    plt.savefig(summary_path)
    logger.info(f"Results summary visualization saved to {summary_path}")
    
    return accuracy

def main():
    """Main execution function for regional bias detection experiment"""
    global model_name
    
    # Parse arguments
    args = parse_arguments()
    model_name = args.model_name
    
    # Create required directories
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.cache_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    
    # Set up logging
    logger = setup_logging(args.log_dir, os.path.basename(model_name).replace('/', '_'))
    
    # Log experimental configuration
    logger.info("Regional Bias Detection Experiment Configuration:")
    for arg, value in vars(args).items():
        # Mask token for security
        if arg == 'hf_token':
            logger.info(f"  {arg}: {'*' * 8 if value else 'Not provided'}")
        else:
            logger.info(f"  {arg}: {value}")
    
    # Set random seed for reproducibility
    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    logger.info(f"Random seed set to {args.random_seed}")
    
    # Start timing
    start_time = time.time()
    
    # Load datasets
    examples_df, test_df = load_datasets(
        args.examples_path, args.test_path, logger, args.test_limit
    )
    
    # Set up model and tokenizer
    model, tokenizer, device = setup_model(
        args.model_name, args.cache_dir, args.gpu_id, args.hf_token, logger
    )
    
    # Process test dataset with few-shot learning
    logger.info(f"Processing {len(test_df)} comments with all {len(examples_df)} few-shot examples...")
    
    predictions, raw_outputs = batch_predict(
        model, tokenizer, test_df, examples_df, device,
        random_seed=args.random_seed,
        max_length=args.max_length,
        max_tokens=args.max_tokens,
        checkpoint_interval=args.checkpoint_interval,
        output_dir=args.output_dir,
        logger=logger
    )
    
    # Save results and generate visualizations
    accuracy = save_results(
        test_df, predictions, raw_outputs, args.output_dir, logger, args.model_name
    )
    
    # Report execution time
    end_time = time.time()
    elapsed_hours = (end_time - start_time) / 3600
    logger.info(f"Total execution time: {elapsed_hours:.2f} hours")
    
    # Final summary
    logger.info("===== Final Summary =====")
    logger.info(f"Model: {args.model_name}")
    logger.info(f"Test set size: {len(test_df)}")
    logger.info(f"Few-shot examples: {len(examples_df)}")
    logger.info(f"Accuracy: {accuracy:.4f}")

if __name__ == "__main__":
    main()
