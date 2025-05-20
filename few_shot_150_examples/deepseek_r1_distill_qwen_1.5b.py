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
from tqdm import tqdm
from huggingface_hub import login

# Default configurations for the model
DEFAULT_MODEL_NAME = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
DEFAULT_GPU_ID = 0
DEFAULT_RANDOM_SEED = 42
DEFAULT_MAX_LENGTH = 4096
DEFAULT_MAX_TOKENS = 50

def parse_arguments():
    """Set up and parse command line arguments for the bias detection pipeline"""
    parser = argparse.ArgumentParser(description='Few-shot learning for regional bias detection with careful reasoning')
    
    # Dataset paths - set these to your dataset locations
    parser.add_argument('--examples_path', type=str, 
                        default=os.environ.get('EXAMPLES_PATH', 'data/150_examples_few_shot_classification_dataset.csv'),
                        help='Path to CSV file with few-shot examples')
    
    parser.add_argument('--test_path', type=str, 
                        default=os.environ.get('TEST_PATH', 'data/annotated_dataset.csv'),
                        help='Path to CSV file with test dataset')
    
    # Output directories - set these to your preferred locations
    parser.add_argument('--output_dir', type=str, 
                        default=os.environ.get('OUTPUT_DIR', 'results/deepseek_few_shot_150'),
                        help='Directory to save results')
    
    parser.add_argument('--cache_dir', type=str, 
                        default=os.environ.get('CACHE_DIR', 'model_cache'),
                        help='Directory for model cache')
    
    parser.add_argument('--log_dir', type=str, 
                        default=os.environ.get('LOG_DIR', 'logs'),
                        help='Directory for log files')
    
    # Model settings
    parser.add_argument('--model_name', type=str, 
                        default=os.environ.get('MODEL_NAME', DEFAULT_MODEL_NAME),
                        help='Model name or path')
    
    parser.add_argument('--gpu_id', type=int, 
                        default=int(os.environ.get('GPU_ID', DEFAULT_GPU_ID)),
                        help='GPU ID to use')
    
    parser.add_argument('--hf_token', type=str, 
                        default=os.environ.get('HF_TOKEN', ''),
                        help='HuggingFace token (required - set via env var or command line)')
    
    # Experiment settings
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
    
    parser.add_argument('--num_examples', type=int, 
                        default=150,
                        help='Number of few-shot examples to use (default: 150)')
    
    parser.add_argument('--slow_mode', action='store_true',
                        help='Enable slow mode with pauses between examples for better quality')
    
    parser.add_argument('--float16', action='store_true',
                        help='Use float16 precision instead of quantization')
    
    return parser.parse_args()

def create_directory(directory_path, logger=None):
    """Simple function to create a directory if it doesn't exist"""
    os.makedirs(directory_path, exist_ok=True)
    if logger:
        logger.info(f"Directory created/verified: {directory_path}")

def setup_logging(log_dir, model_name):
    """Configure logging to both file and console"""
    # Create log directory
    create_directory(log_dir)
    
    # Create a timestamped log file
    log_file = os.path.join(log_dir, f"{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    
    # Set up logging configuration
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    return logging.getLogger(__name__)

def load_datasets(examples_path, test_path, logger, num_examples=150, test_limit=None, random_seed=42):
    """Load and prepare the few-shot examples and test dataset"""
    # Verify files exist
    if not os.path.exists(examples_path):
        logger.error(f"Examples file not found: {examples_path}")
        raise FileNotFoundError(f"Examples file not found: {examples_path}")
        
    if not os.path.exists(test_path):
        logger.error(f"Test dataset file not found: {test_path}")
        raise FileNotFoundError(f"Test dataset file not found: {test_path}")
    
    # Load the datasets
    logger.info(f"Loading examples from {examples_path}")
    examples_df = pd.read_csv(examples_path)
    
    logger.info(f"Loading test dataset from {test_path}")
    test_df = pd.read_csv(test_path)
    
    logger.info(f"Loaded {len(examples_df)} examples and {len(test_df)} test comments")
    
    # Select a balanced subset of examples if requested
    examples_per_class = num_examples // 2
    if len(examples_df) > num_examples:
        # Get positive examples (regional bias)
        pos_examples = examples_df[examples_df['Level-1'] >= 1].sample(examples_per_class, random_state=random_seed)
        # Get negative examples (non-regional bias)
        neg_examples = examples_df[examples_df['Level-1'] == 0].sample(examples_per_class, random_state=random_seed)
        # Combine them
        selected_examples = pd.concat([pos_examples, neg_examples])
        logger.info(f"Selected balanced subset of {len(selected_examples)} examples for few-shot learning")
    else:
        selected_examples = examples_df
        logger.warning(f"Requested {num_examples} examples but only {len(examples_df)} are available")
    
    # Ensure no overlap between examples and test data
    examples_comments = set(selected_examples['Comment'].str.strip())
    test_df = test_df[~test_df['Comment'].str.strip().isin(examples_comments)]
    logger.info(f"After removing overlapping comments, {len(test_df)} test comments remain")
    
    # Apply test limit if specified
    if test_limit is not None and test_limit > 0:
        logger.info(f"Limiting test set to {test_limit} examples")
        test_df = test_df.head(test_limit)
    
    return selected_examples, test_df

def prepare_prompt(comment, examples_df):
    """Create a detailed prompt with examples and step-by-step reasoning instructions"""
    # Base instruction explaining the task
    base_prompt = """You are an expert in identifying regional biases in comments about Indian states and regions. 
Your task is to determine whether a comment contains regional bias or not.
Regional bias includes stereotypes, prejudices, or discriminatory statements about:
- Indian states or regions
- People from specific Indian states  
- Cultural, linguistic, economic, political, or infrastructural aspects of Indian regions

IMPORTANT: Take your time and consider each comment very carefully. Think step-by-step.

Here are some examples of comments and their classifications:

"""
    
    # Add examples from our dataset
    for _, row in examples_df.iterrows():
        example_text = row['Comment']
        classification = "regional_bias" if row['Level-1'] >= 1 else "non_regional_bias"
        base_prompt += f"Comment: \"{example_text}\"\nClassification: {classification}\n\n"
    
    # Add the current comment to classify with detailed instructions
    base_prompt += f"""Now, please analyze the following comment VERY CAREFULLY step by step:
Comment: "{comment}"

Step 1: First, identify if this comment mentions any Indian state, region, or people from specific regions.
(Write your analysis for Step 1)

Step 2: Check if the comment contains any of these elements:
- Stereotypical statements about people from a region
- Generalizations about a state or its people
- Discriminatory language targeting regional identity
- Prejudiced views about regional culture, language, or traditions
- Biased statements about economic or developmental status
- Political stereotypes associated with regions
(Write your analysis for Step 2)

Step 3: Determine if these elements, if present, constitute bias or are merely factual/neutral observations.
(Write your analysis for Step 3)

Step 4: Based on your thorough analysis above, provide your final classification.
Think carefully before deciding. If you identify any stereotypes, generalizations, or prejudice about an Indian state or region, classify as "regional_bias".
If it's neutral, factual, or does not contain regional bias, classify as "non_regional_bias".

Classification (ONLY respond with exactly "regional_bias" or "non_regional_bias" as your final answer):"""
    
    return base_prompt

def setup_model(model_name, cache_dir, gpu_id, hf_token, logger, use_float16=False):
    """Load and configure the language model for inference"""
    # Create cache directory
    create_directory(cache_dir, logger)
    
    # Set environment variables for caching
    os.environ["TRANSFORMERS_CACHE"] = cache_dir
    os.environ["HF_HOME"] = cache_dir
    os.environ["HF_DATASETS_CACHE"] = cache_dir
    logger.info(f"Using cache directory: {cache_dir}")
    
    # Configure GPU
    if torch.cuda.is_available():
        os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
        torch.cuda.set_device(0)  # After setting CUDA_VISIBLE_DEVICES, we use device 0
        device = torch.device("cuda:0")
        logger.info(f"Using GPU {gpu_id}: {torch.cuda.get_device_name(0)}")
        logger.info(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    else:
        device = torch.device("cpu")
        logger.info("CUDA not available. Using CPU.")
    
    # Clear GPU memory before loading model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # Login to HuggingFace if token provided
    if hf_token:
        login(token=hf_token)
        logger.info("Logged in to HuggingFace")
    
    logger.info(f"Loading model: {model_name}")
    start_time = time.time()
    
    # Configure tokenizer options
    tokenizer_kwargs = {}
    if hf_token:
        tokenizer_kwargs['token'] = hf_token
    if cache_dir:
        tokenizer_kwargs['cache_dir'] = cache_dir
    
    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, **tokenizer_kwargs)
    
    # Ensure tokenizer has padding token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Configure model loading options
    model_kwargs = {'low_cpu_mem_usage': True}
    
    if hf_token:
        model_kwargs['token'] = hf_token
    if cache_dir:
        model_kwargs['cache_dir'] = cache_dir
    
    if torch.cuda.is_available():
        model_kwargs['device_map'] = "auto"
    
    # Choose precision based on settings
    if use_float16:
        # Use float16 precision for better quality
        logger.info("Using float16 precision for better quality")
        model_kwargs['torch_dtype'] = torch.float16
    else:
        # Use 8-bit quantization for memory efficiency
        logger.info("Using 8-bit quantization for memory efficiency")
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=True,
            bnb_8bit_use_double_quant=True,
            bnb_8bit_compute_dtype=torch.float16
        )
        model_kwargs['quantization_config'] = quantization_config
    
    # Load the model
    model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
    
    # Set to evaluation mode
    model.eval()
    
    elapsed_time = time.time() - start_time
    logger.info(f"Model loaded in {elapsed_time:.2f} seconds")
    
    return model, tokenizer, device

def predict_with_model(model, tokenizer, prompt, device, max_length=4096, max_tokens=50, logger=None):
    """Generate prediction using the model with careful reasoning"""
    # Check if prompt is too long
    tokenized_prompt = tokenizer.encode(prompt)
    if len(tokenized_prompt) > max_length and logger:
        logger.warning(f"Prompt has {len(tokenized_prompt)} tokens, exceeding max_length of {max_length}. Will truncate.")
    
    # Tokenize the prompt with truncation
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_length).to(device)
    
    # Log token count if in debug mode
    if logger and logger.isEnabledFor(logging.DEBUG):
        logger.debug(f"Tokenized prompt length: {inputs['input_ids'].shape[1]} tokens")
    
    # Generate response
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            pad_token_id=tokenizer.eos_token_id,
            do_sample=False,  # Deterministic generation
            temperature=0.1,  # Very low temperature for more decisive outputs
            num_beams=4,      # Use beam search for more careful consideration
            early_stopping=True  # Stop when a good answer is found
        )
    
    # Decode the generated text
    full_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract only the last part with the classification
    prompt_end = full_output.find("Classification (ONLY respond with")
    if prompt_end != -1:
        # Get text after the prompt
        response = full_output[prompt_end:].lower()
    else:
        # Otherwise just look at the end of the output
        response = full_output[-100:].lower()
    
    # Free memory
    del inputs, outputs
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # Extract the classification from the response
    if "regional_bias" in response and "non_regional_bias" not in response:
        return 1, full_output  # Clearly regional_bias
    elif "non_regional_bias" in response:
        return 0, full_output  # Clearly non_regional_bias
    elif "regional" in response and "bias" in response and "non" not in response:
        return 1, full_output  # Likely regional_bias
    else:
        # Look more broadly in the full output for hints
        if any(term in full_output.lower() for term in ["stereotype", "prejudice", "discriminat"]):
            return 1, full_output  # Likely regional_bias based on reasoning
        else:
            return 0, full_output  # Default to non_regional_bias if unclear

def batch_predict(model, tokenizer, test_df, examples_df, device, slow_mode=False, max_length=4096, 
                max_tokens=50, checkpoint_interval=10, output_dir=None, logger=None):
    """Process all test comments and make predictions"""
    predictions = []
    raw_outputs = []
    
    # Get test comments
    test_comments = test_df["Comment"].tolist()
    
    # Create checkpoint directory if needed
    if output_dir:
        checkpoint_dir = os.path.join(output_dir, "checkpoints")
        create_directory(checkpoint_dir, logger)
    
    # Estimate total time
    estimated_time = len(test_comments) / 60 / 60  # hours
    logger.info(f"Processing {len(test_comments)} comments...")
    if slow_mode:
        logger.info("Slow mode enabled - adding pauses for better quality")
        logger.info(f"Estimated time: ~{estimated_time:.1f} hours")
    
    # Process each test comment
    for i, comment in enumerate(tqdm(test_comments, desc="Processing examples")):
        # Add artificial slow-down in slow mode
        if slow_mode:
            time.sleep(1)  # Add 1 second pause between examples
        
        # Generate prompt for this comment
        prompt = prepare_prompt(comment, examples_df)
        
        # Get prediction
        try:
            prediction, raw_output = predict_with_model(
                model, tokenizer, prompt, device, max_length, max_tokens, logger
            )
        except Exception as e:
            logger.error(f"Error processing comment {i+1}: {e}")
            prediction = 0  # Default to non-regional bias
            raw_output = f"ERROR: {str(e)}"
        
        # Store results
        predictions.append(prediction)
        raw_outputs.append(raw_output)
        
        # Log progress
        if (i + 1) % 10 == 0 or i == 0:
            logger.info(f"Processed example {i+1}/{len(test_comments)}")
            logger.info(f"Decision: {prediction} (0=non-regional, 1=regional)")
            
            # Log a snippet of the output for debugging
            output_snippet = raw_output[-150:] if len(raw_output) > 150 else raw_output
            logger.info(f"Last part of output: {output_snippet}")
        
        # Save checkpoint if enabled
        if output_dir and ((i + 1) % checkpoint_interval == 0 or i == len(test_comments) - 1):
            # Create a short model name for file naming
            model_short_name = os.path.basename(model_name).replace('/', '_')
            
            checkpoint_df = pd.DataFrame({
                'Comment': test_df['Comment'].iloc[:i+1].tolist(),
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
    """Save prediction results and evaluation metrics"""
    # Get true labels (1 if Level-1 >= 1, otherwise 0)
    true_labels = test_df['Level-1'].apply(lambda x: 1 if x >= 1 else 0).tolist()
    
    # Create visualization directory
    viz_dir = os.path.join(output_dir, "visualizations")
    create_directory(viz_dir, logger)
    
    # Save predictions with raw outputs
    results_df = test_df.copy()
    results_df['Predicted'] = predictions
    
    # Truncate raw outputs to prevent huge files
    truncated_outputs = [str(output)[:500] for output in raw_outputs]
    results_df['Model_Output'] = truncated_outputs
    
    # Create timestamped output paths
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
    
    # Calculate accuracy
    accuracy = accuracy_score(true_labels, predictions)
    logger.info(f"Accuracy: {accuracy:.4f}")
    
    # Generate confusion matrix visualization
    plt.figure(figsize=(10, 8))
    cm = confusion_matrix(true_labels, predictions)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Non-Regional Bias', 'Regional Bias'],
                yticklabels=['Non-Regional Bias', 'Regional Bias'])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(f'Confusion Matrix - {model_short_name}')
    plt.tight_layout()
    plt.savefig(matrix_path)
    logger.info(f"Confusion matrix saved to {matrix_path}")
    
    # Create comprehensive results visualization with 2x2 grid
    plt.figure(figsize=(12, 8))
    
    # Plot 1: Confusion matrix as heatmap
    plt.subplot(2, 2, 1)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Non-Regional', 'Regional'],
                yticklabels=['Non-Regional', 'Regional'])
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    
    # Plot 2: Class distribution
    plt.subplot(2, 2, 2)
    class_counts = pd.Series(true_labels).value_counts().sort_index()
    plt.bar(['Non-Regional', 'Regional'], [class_counts.get(0, 0), class_counts.get(1, 0)], 
            color=['#1f77b4', '#ff7f0e'])
    plt.title('Test Set Class Distribution')
    plt.ylabel('Number of Samples')
    
    # Add value labels on bars
    for i, v in enumerate([class_counts.get(0, 0), class_counts.get(1, 0)]):
        plt.text(i, v + 5, str(v), ha='center')
    
    # Plot 3: Prediction distribution
    plt.subplot(2, 2, 3)
    pred_counts = pd.Series(predictions).value_counts().sort_index()
    plt.bar(['Non-Regional', 'Regional'], [pred_counts.get(0, 0), pred_counts.get(1, 0)], 
            color=['#2ca02c', '#d62728'])
    plt.title('Model Predictions')
    plt.ylabel('Number of Samples')
    
    # Add value labels on bars
    for i, v in enumerate([pred_counts.get(0, 0), pred_counts.get(1, 0)]):
        plt.text(i, v + 5, str(v), ha='center')
    
    # Plot 4: Accuracy
    plt.subplot(2, 2, 4)
    plt.bar(['Accuracy'], [accuracy], color='#9467bd')
    plt.ylim(0, 1.0)
    plt.title('Model Performance')
    plt.text(0, accuracy + 0.05, f"{accuracy:.4f}", ha='center')
    
    plt.tight_layout()
    plt.savefig(summary_path)
    logger.info(f"Results summary visualization saved to {summary_path}")
    
    return accuracy

def main():
    """Main execution function for the regional bias detection pipeline"""
    global args, model_name
    
    # Parse command line arguments
    args = parse_arguments()
    model_name = args.model_name
    
    # Check HuggingFace token
    if args.hf_token:
        print(f"HuggingFace token is set and will be used for model loading")
    else:
        print(f"WARNING: No HuggingFace token provided - set via HF_TOKEN env var or --hf_token argument")
    
    # Create required directories
    for directory in [args.output_dir, args.cache_dir, args.log_dir]:
        create_directory(directory)
    
    # Create a short model name for file naming
    model_short_name = os.path.basename(args.model_name).replace('/', '_')
    
    # Set up logging
    logger = setup_logging(args.log_dir, model_short_name)
    
    # Log all arguments
    logger.info("Arguments:")
    for arg, value in vars(args).items():
        # Don't log the token for security
        if arg == 'hf_token':
            logger.info(f"  {arg}: {'*' * 8 if value else 'Not provided'}")
        else:
            logger.info(f"  {arg}: {value}")
    
    # Set random seed for reproducibility
    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    logger.info(f"Random seed set to {args.random_seed}")
    
    # Start timing the execution
    start_time = time.time()
    
    # Load the few-shot examples and test datasets
    examples_df, test_df = load_datasets(
        args.examples_path, args.test_path, logger, 
        num_examples=args.num_examples,
        test_limit=args.test_limit,
        random_seed=args.random_seed
    )
    
    # Set up model and tokenizer
    model, tokenizer, device = setup_model(
        args.model_name, args.cache_dir, args.gpu_id, args.hf_token, logger,
        use_float16=args.float16
    )
    
    # Make predictions on the test set
    logger.info(f"Processing {len(test_df)} comments with {len(examples_df)} few-shot examples...")
    
    predictions, raw_outputs = batch_predict(
        model, tokenizer, test_df, examples_df, device,
        slow_mode=args.slow_mode,
        max_length=args.max_length,
        max_tokens=args.max_tokens,
        checkpoint_interval=args.checkpoint_interval,
        output_dir=args.output_dir,
        logger=logger
    )
    
    # Save results and visualizations
    accuracy = save_results(
        test_df, predictions, raw_outputs, args.output_dir, logger, args.model_name
    )
    
    # Calculate total execution time
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
