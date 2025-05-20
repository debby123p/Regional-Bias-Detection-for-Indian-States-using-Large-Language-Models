

import os
import sys
import pandas as pd
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
import time
import logging
import argparse
from datetime import datetime
from huggingface_hub import login
import re
import gc

# --- Default Configurations ---
# These can be overridden by command-line arguments or environment variables.
DEFAULT_MODEL_NAME = 'Qwen/Qwen2.5-7B-Instruct'
DEFAULT_GPU_ID = 1  # Note: This script defaults to GPU 1.
DEFAULT_RANDOM_SEED = 42
DEFAULT_MAX_LENGTH = 4096  # Context window for 150 examples.

def parse_arguments():
    """Parse command line arguments with sensible defaults."""
    parser = argparse.ArgumentParser(description='Few-shot learning for regional bias detection with Qwen2.5 (150 examples)')

    # --- Path Configurations ---
    # Ensure these paths point to your specific file locations or set corresponding environment variables.

    # Path to the CSV file containing the 150 few-shot examples.
    # Default: 'data/150_examples_few_shot_classification_dataset.csv'
    parser.add_argument('--examples_path', type=str,
                        default=os.environ.get('EXAMPLES_PATH', 'data/150_examples_few_shot_classification_dataset.csv'),
                        help='Path to CSV file with 150 few-shot examples')

    # Path to the CSV file containing the main test dataset.
    # Default: 'data/annotated_dataset.csv'
    parser.add_argument('--test_path', type=str,
                        default=os.environ.get('TEST_PATH', 'data/annotated_dataset.csv'),
                        help='Path to CSV file with test dataset')

    # Directory where results (predictions, reports, visualizations) will be saved.
    # Default: 'results/qwen_few_shot_150'
    parser.add_argument('--output_dir', type=str,
                        default=os.environ.get('OUTPUT_DIR', 'results/qwen_few_shot_150'),
                        help='Directory to save results')

    # Directory for caching downloaded HuggingFace models and tokenizers.
    # Default: 'model_cache'
    parser.add_argument('--cache_dir', type=str,
                        default=os.environ.get('CACHE_DIR', 'model_cache'),
                        help='Directory for model cache')

    # Directory where log files will be stored.
    # Default: 'logs'
    parser.add_argument('--log_dir', type=str,
                        default=os.environ.get('LOG_DIR', 'logs'),
                        help='Directory for log files')

    # --- Model and Execution Configurations ---
    parser.add_argument('--model_name', type=str,
                        default=os.environ.get('MODEL_NAME', DEFAULT_MODEL_NAME),
                        help='HuggingFace model name or local path')
    parser.add_argument('--gpu_id', type=int,
                        default=int(os.environ.get('GPU_ID', DEFAULT_GPU_ID)),
                        help='GPU ID to use (e.g., 0, 1). Defaults to GPU 1.')
    parser.add_argument('--hf_token', type=str,
                        default=os.environ.get('HF_TOKEN', ''),
                        help='HuggingFace API token (optional, recommended to use environment variable HF_TOKEN)')
    parser.add_argument('--random_seed', type=int,
                        default=int(os.environ.get('RANDOM_SEED', DEFAULT_RANDOM_SEED)),
                        help='Random seed for reproducibility')
    parser.add_argument('--test_limit', type=int, default=None,
                        help='Limit the number of test examples for quick runs (e.g., for debugging)')
    parser.add_argument('--checkpoint_interval', type=int, default=10,
                        help='Interval (number of examples) for saving intermediate prediction checkpoints')
    parser.add_argument('--max_length', type=int,
                        default=int(os.environ.get('MAX_LENGTH', DEFAULT_MAX_LENGTH)),
                        help='Maximum sequence length for model tokenization')

    return parser.parse_args()

def create_directory(directory_path, logger=None):
    """Creates a directory if it doesn't already exist."""
    os.makedirs(directory_path, exist_ok=True)
    if logger:
        logger.info(f"Ensured directory exists: {directory_path}")

def setup_logging(log_dir, model_name_prefix):
    """Set up logging to file and console."""
    create_directory(log_dir) # Ensure log directory exists
    log_file_name = f"{model_name_prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    log_file_path = os.path.join(log_dir, log_file_name)

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file_path),
            logging.StreamHandler(sys.stdout) # Also log to console
        ]
    )
    logger = logging.getLogger(__name__)
    logger.info(f"Logging initialized. Log file: {log_file_path}")
    return logger

def clean_text(text):
    """Clean and normalize text for model input."""
    text = str(text).lower()
    text = re.sub(r"http\S+", "", text)  # Remove URLs
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)  # Remove special characters (keeps alphanumeric and spaces)
    text = re.sub(r"\s+", " ", text).strip()  # Normalize whitespace
    return text

def load_datasets(examples_path, test_path, logger, test_limit=None):
    """Load and preprocess the few-shot examples and test datasets."""
    if not os.path.exists(examples_path):
        logger.error(f"Examples file not found: {examples_path}")
        raise FileNotFoundError(f"Examples file not found: {examples_path}")
    if not os.path.exists(test_path):
        logger.error(f"Test dataset file not found: {test_path}")
        raise FileNotFoundError(f"Test dataset file not found: {test_path}")

    logger.info(f"Loading few-shot examples from: {examples_path}")
    examples_df = pd.read_csv(examples_path)
    logger.info(f"Loaded {len(examples_df)} few-shot examples.")

    logger.info(f"Loading test dataset from: {test_path}")
    test_df = pd.read_csv(test_path)
    logger.info(f"Loaded {len(test_df)} total test comments.")

    # Verify class distribution in examples (expecting 75 of each for the 150_examples dataset)
    bias_examples_count = len(examples_df[examples_df['Level-1'] >= 1])
    non_bias_examples_count = len(examples_df[examples_df['Level-1'] == 0])
    logger.info(f"Found {bias_examples_count} regional bias examples and {non_bias_examples_count} non-regional bias examples in the few-shot set.")
    
    expected_count_per_class = 75
    if bias_examples_count != expected_count_per_class or non_bias_examples_count != expected_count_per_class:
        logger.warning(
            f"Expected {expected_count_per_class} examples of each class in the few-shot set, "
            f"but found {bias_examples_count} bias and {non_bias_examples_count} non-bias examples. "
            f"Ensure this is intended for your '{os.path.basename(examples_path)}' file."
        )

    # Ensure no overlap between few-shot examples and test data based on 'Comment'
    examples_comments_set = set(examples_df['Comment'].astype(str).str.strip())
    original_test_len = len(test_df)
    test_df = test_df[~test_df['Comment'].astype(str).str.strip().isin(examples_comments_set)]
    if len(test_df) < original_test_len:
        logger.info(f"Removed {original_test_len - len(test_df)} comments from test set that were present in the examples set. "
                    f"{len(test_df)} test comments remain.")
    else:
        logger.info("No overlap found between examples and test set comments.")

    logger.info("Cleaning comment text for both datasets...")
    test_df["Cleaned_Comment"] = test_df["Comment"].apply(clean_text)
    examples_df["Cleaned_Comment"] = examples_df["Comment"].apply(clean_text)

    if test_limit is not None and test_limit > 0:
        logger.info(f"Limiting test set to the first {test_limit} examples as per 'test_limit' argument.")
        test_df = test_df.head(test_limit)

    return examples_df, test_df

def create_few_shot_prompt(examples_df, comment_to_classify, random_seed=42):
    """
    Construct a few-shot prompt using all 150 examples, shuffled for diversity.
    """
    shuffled_examples = examples_df.sample(frac=1, random_state=random_seed)

    prompt_parts = [
        "You are an expert in identifying regional biases in comments about Indian states and regions.",
        "Task: Classify if the comment contains regional bias related to Indian states or regions.\n",
        "Instructions:",
        "- Regional Bias (1): Comments that contain stereotypes, prejudices, or biases about specific Indian states or regions.",
        "- Non-Regional Bias (0): Comments that don't contain regional stereotypes or biases about Indian states.\n",
        "Examples:"
    ]
    
    for _, row in shuffled_examples.iterrows():
        classification_label = 1 if row['Level-1'] >= 1 else 0
        prompt_parts.append(f"Comment: \"{row['Cleaned_Comment']}\"\nClassification: {classification_label}\n")
    
    prompt_parts.append(f"Now classify this comment:\n\"{comment_to_classify}\"\nClassification:")
    
    return "\n".join(prompt_parts)

def setup_model(model_name, cache_dir, gpu_id, hf_token, logger):
    """Load the HuggingFace model and tokenizer with specified configurations."""
    create_directory(cache_dir, logger) # Ensure cache directory exists

    # Set HuggingFace environment variables for caching
    os.environ["TRANSFORMERS_CACHE"] = cache_dir
    os.environ["HF_HOME"] = cache_dir
    os.environ["HF_DATASETS_CACHE"] = cache_dir
    logger.info(f"HuggingFace cache directory set to: {cache_dir}")

    if torch.cuda.is_available():
        # Setting CUDA_VISIBLE_DEVICES scopes GPU visibility for this process
        os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
        # PyTorch then sees the chosen GPU as device 'cuda:0'
        device = torch.device("cuda:0")
        torch.cuda.set_device(device) # Explicitly set the device
        logger.info(f"Using GPU {gpu_id} ({torch.cuda.get_device_name(0)}). Visible GPUs: {os.environ['CUDA_VISIBLE_DEVICES']}")
        logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        torch.cuda.empty_cache() # Clear any cached memory
    else:
        device = torch.device("cpu")
        logger.info("CUDA not available. Using CPU.")

    if hf_token:
        try:
            login(token=hf_token)
            logger.info("Successfully logged in to HuggingFace Hub.")
        except Exception as e:
            logger.warning(f"HuggingFace login failed. Proceeding without login. Error: {e}")

    logger.info(f"Loading model: {model_name}")
    load_start_time = time.time()

    try:
        # Tokenizer configuration - Qwen models may require trust_remote_code
        tokenizer_kwargs = {'trust_remote_code': True}
        if hf_token: tokenizer_kwargs['token'] = hf_token
        if cache_dir: tokenizer_kwargs['cache_dir'] = cache_dir
        tokenizer = AutoTokenizer.from_pretrained(model_name, **tokenizer_kwargs)

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            logger.info("Tokenizer pad_token was None, set to eos_token.")

        # Quantization configuration for memory efficiency (8-bit for this scenario)
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=True,
            bnb_8bit_use_double_quant=True,
            bnb_8bit_compute_dtype=torch.float16 # Use float16 for computation with 8-bit weights
        )

        # Model loading configuration - Qwen models may require trust_remote_code
        model_kwargs = {
            'quantization_config': quantization_config,
            'low_cpu_mem_usage': True, # Optimizes memory on CPU during loading
            'trust_remote_code': True,
        }
        if hf_token: model_kwargs['token'] = hf_token
        if cache_dir: model_kwargs['cache_dir'] = cache_dir
        if torch.cuda.is_available(): model_kwargs['device_map'] = "auto" # Automatically map model layers to available GPUs

        model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
        model.eval() # Set model to evaluation mode

        load_elapsed_time = time.time() - load_start_time
        logger.info(f"Model and tokenizer loaded successfully in {load_elapsed_time:.2f} seconds.")
        return model, tokenizer, device

    except Exception as e:
        logger.error(f"Fatal error during model loading: {e}", exc_info=True)
        raise

def predict_with_model(model, tokenizer, prompt, device, max_context_length=4096, max_new_tokens=10, logger=None):
    """Generate a classification (0 or 1) from the model given a prompt."""
    try:
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_context_length).to(device)
        
        if logger and logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"Tokenized prompt length for prediction: {inputs['input_ids'].shape[1]} tokens.")

        with torch.no_grad(): # Disable gradient calculations for inference
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens, # Limit the number of tokens generated for the classification
                temperature=0.1,  # Lower temperature for more deterministic output
                do_sample=False,   # Disable sampling for deterministic output
                num_beams=1,       # Use greedy decoding
                pad_token_id=tokenizer.eos_token_id # Important for Qwen and other models
            )
        
        full_output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract only the newly generated text part
        # The prompt itself is part of inputs['input_ids'], so decode outputs[0] and slice
        input_length = inputs['input_ids'].shape[1]
        generated_ids = outputs[0][input_length:]
        generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

        if logger and logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"Raw generated text: '{generated_text}' (Full output snippet: ...{full_output_text[-100:]})")
        
        del inputs, outputs # Clean up tensors
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # --- Output Parsing Logic ---
        # The model is expected to output '0' or '1'.
        # This logic attempts to robustly parse that from the generated text.
        if generated_text == "1" or generated_text.endswith("1"):
            return 1, full_output_text
        elif generated_text == "0" or generated_text.endswith("0"):
            return 0, full_output_text

        # Fallback: check for keywords if direct '0' or '1' is not found
        # This can be helpful if the model generates more verbose output.
        # Consider the context of "regional bias" vs "non-regional bias" for the keywords.
        lower_generated_text = generated_text.lower()
        if "1" in lower_generated_text and "0" not in lower_generated_text: # Prefer '1' if it's distinctly present
            return 1, full_output_text
        if "0" in lower_generated_text and "1" not in lower_generated_text: # Prefer '0' if it's distinctly present
            return 0, full_output_text
        if "regional bias" in lower_generated_text or "bias" in lower_generated_text: # Check for bias indication
            return 1, full_output_text
        if "non-regional" in lower_generated_text or "no bias" in lower_generated_text: # Check for non-bias indication
            return 0, full_output_text
        
        if logger:
            logger.warning(f"Could not reliably parse '0' or '1' from model output: '{generated_text}'. Defaulting to 0 (Non-Regional Bias).")
        return 0, full_output_text # Default to non-bias if parsing is ambiguous

    except Exception as e:
        if logger:
            logger.error(f"Error during model prediction: {e}", exc_info=True)
        return 0, f"ERROR_DURING_PREDICTION: {str(e)}" # Return default and error message

def batch_predict(model, tokenizer, test_df, examples_df, device, max_context_length=4096, random_seed=42,
                  checkpoint_interval=10, output_dir=None, logger=None, model_file_prefix="model"):
    """Process test comments, generate predictions, and save checkpoints."""
    predictions = []
    raw_model_outputs = []
    test_comments_list = test_df["Cleaned_Comment"].tolist()

    if output_dir:
        checkpoints_subdir = os.path.join(output_dir, "checkpoints")
        create_directory(checkpoints_subdir, logger)

    for i, comment_text in enumerate(test_comments_list):
        try:
            prompt = create_few_shot_prompt(examples_df, comment_text, random_seed)
            
            # Check prompt length before sending to model; Qwen models have specific context limits.
            # This is a simplified check; actual token count depends on the tokenizer.
            # A more robust check would tokenize and count, then potentially truncate examples if too long.
            # For this script, max_length in predict_with_model handles truncation if the full prompt is too long.
            
            # The predict_with_model function already handles truncation based on max_context_length
            predicted_class, raw_output = predict_with_model(
                model, tokenizer, prompt, device, max_context_length, logger=logger
            )
            
            predictions.append(predicted_class)
            raw_model_outputs.append(raw_output)

            if (i + 1) % 10 == 0 or i == 0 or (i + 1) == len(test_comments_list):
                logger.info(f"Processed example {i+1}/{len(test_comments_list)}. Prediction: {predicted_class}")

            # Save checkpoint
            if output_dir and checkpoint_interval > 0 and \
               ((i + 1) % checkpoint_interval == 0 or (i + 1) == len(test_comments_list)):
                
                checkpoint_data = {
                    'Comment': test_df['Comment'].iloc[:i+1].tolist(),
                    'Cleaned_Comment': test_df['Cleaned_Comment'].iloc[:i+1].tolist(),
                    'True_Label': test_df['Level-1'].iloc[:i+1].apply(lambda x: 1 if x >= 1 else 0).tolist(),
                    'Predicted_Label': predictions, # Already sliced by loop
                    'Raw_Model_Output': [str(out)[:1000] for out in raw_model_outputs] # Truncate long outputs for CSV
                }
                checkpoint_df = pd.DataFrame(checkpoint_data)
                checkpoint_file = os.path.join(checkpoints_subdir, f"{model_file_prefix}_checkpoint_{i+1}.csv")
                checkpoint_df.to_csv(checkpoint_file, index=False)
                logger.info(f"Saved checkpoint for {i+1} examples to {checkpoint_file}")

        except Exception as e:
            logger.error(f"Error processing comment at index {i} ('{comment_text[:50]}...'): {e}", exc_info=True)
            predictions.append(0) # Default prediction on error
            raw_model_outputs.append(f"ERROR_IN_BATCH_PROCESSING: {str(e)}")
        
        # Aggressive cache clearing for large models and long contexts
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
            
    return predictions, raw_model_outputs

def save_results(test_df, predictions, raw_outputs, output_dir, logger, model_file_prefix="qwen_7b_150examples"):
    """Save final predictions, evaluation report, and visualizations."""
    true_labels = test_df['Level-1'].apply(lambda x: 1 if x >= 1 else 0).tolist()

    visualizations_subdir = os.path.join(output_dir, "visualizations")
    create_directory(visualizations_subdir, logger)

    # Prepare results DataFrame
    results_df = test_df.copy()
    results_df['Predicted_Label'] = predictions
    results_df['True_Label_Binary'] = true_labels
    # Truncate raw outputs for manageable CSV files
    results_df['Raw_Model_Output'] = [str(output)[:1000] for output in raw_outputs] 

    current_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    predictions_file = os.path.join(output_dir, f"{model_file_prefix}_all_predictions_{current_timestamp}.csv")
    report_file = os.path.join(output_dir, f"{model_file_prefix}_classification_report_{current_timestamp}.txt")
    confusion_matrix_file = os.path.join(visualizations_subdir, f"{model_file_prefix}_confusion_matrix_{current_timestamp}.png")
    summary_plot_file = os.path.join(visualizations_subdir, f"{model_file_prefix}_results_summary_plot_{current_timestamp}.png")

    results_df.to_csv(predictions_file, index=False)
    logger.info(f"All predictions saved to: {predictions_file}")

    # Classification report
    class_report_str = classification_report(true_labels, predictions, target_names=['Non-Regional Bias (0)', 'Regional Bias (1)'])
    accuracy = accuracy_score(true_labels, predictions)
    f1 = f1_score(true_labels, predictions, average='binary') # Binary F1 for the positive class (Regional Bias)

    with open(report_file, 'w') as f:
        f.write(f"Classification Report for: {model_file_prefix}\n")
        f.write(f"Timestamp: {current_timestamp}\n\n")
        f.write(f"Accuracy: {accuracy:.4f}\n")
        f.write(f"F1 Score (Regional Bias): {f1:.4f}\n\n")
        f.write(class_report_str)
    logger.info(f"Classification report saved to: {report_file}")
    logger.info(f"Overall Accuracy: {accuracy:.4f}")
    logger.info(f"Overall F1 Score (Regional Bias): {f1:.4f}")

    # Confusion Matrix
    cm = confusion_matrix(true_labels, predictions)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Predicted Non-Bias', 'Predicted Bias'],
                yticklabels=['Actual Non-Bias', 'Actual Bias'])
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title(f'Confusion Matrix - {model_file_prefix}')
    plt.tight_layout()
    plt.savefig(confusion_matrix_file)
    logger.info(f"Confusion matrix saved to: {confusion_matrix_file}")
    plt.close()

    # Comprehensive Results Summary Plot
    fig, axs = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle(f'Results Summary - {model_file_prefix}', fontsize=16)

    # Confusion Matrix (again, for the summary plot)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axs[0,0],
                xticklabels=['Non-Bias', 'Bias'], yticklabels=['Non-Bias', 'Bias'])
    axs[0,0].set_title('Confusion Matrix')
    axs[0,0].set_xlabel('Predicted')
    axs[0,0].set_ylabel('True')

    # True Class Distribution
    true_counts = pd.Series(true_labels).value_counts().sort_index()
    axs[0,1].bar(['Non-Bias (0)', 'Bias (1)'], 
                 [true_counts.get(0,0), true_counts.get(1,0)], 
                 color=['skyblue', 'salmon'])
    axs[0,1].set_title('Test Set: True Class Distribution')
    axs[0,1].set_ylabel('Number of Samples')
    for i, count in enumerate([true_counts.get(0,0), true_counts.get(1,0)]):
        axs[0,1].text(i, count + 5, str(count), ha='center')
    
    # Predicted Class Distribution
    pred_counts = pd.Series(predictions).value_counts().sort_index()
    axs[1,0].bar(['Non-Bias (0)', 'Bias (1)'], 
                 [pred_counts.get(0,0), pred_counts.get(1,0)], 
                 color=['lightgreen', 'gold'])
    axs[1,0].set_title('Model Predictions: Class Distribution')
    axs[1,0].set_ylabel('Number of Samples')
    for i, count in enumerate([pred_counts.get(0,0), pred_counts.get(1,0)]):
        axs[1,0].text(i, count + 5, str(count), ha='center')

    # Key Metrics
    metrics_names = ['Accuracy', 'F1 Score (Bias)']
    metrics_values = [accuracy, f1]
    axs[1,1].bar(metrics_names, metrics_values, color=['mediumpurple', 'lightcoral'])
    axs[1,1].set_ylim(0, 1.05)
    axs[1,1].set_title('Key Performance Metrics')
    for i, value in enumerate(metrics_values):
        axs[1,1].text(i, value + 0.02, f"{value:.4f}", ha='center')

    plt.tight_layout(rect=[0, 0, 1, 0.96]) # Adjust layout to make space for suptitle
    plt.savefig(summary_plot_file)
    logger.info(f"Results summary plot saved to: {summary_plot_file}")
    plt.close()

    return accuracy, f1

def main():
    """Main execution script for few-shot regional bias detection."""
    args = parse_arguments()

    # Define a consistent file prefix based on the model name for output files
    # Simplifies model name like 'Qwen/Qwen2.5-7B-Instruct' to 'Qwen2.5-7B-Instruct_150examples'
    base_model_name = args.model_name.split('/')[-1] if '/' in args.model_name else args.model_name
    model_file_prefix = f"{base_model_name}_150examples_fs" # fs for few-shot

    # Setup directories
    for dir_path in [args.output_dir, args.cache_dir, args.log_dir]:
        create_directory(dir_path) # Uses the simplified create_directory

    logger = setup_logging(args.log_dir, model_file_prefix)

    logger.info("--- Starting Few-Shot Regional Bias Detection ---")
    logger.info("Script Arguments:")
    for arg, value in vars(args).items():
        if arg == 'hf_token' and value:
            logger.info(f"  {arg}: **** (Token is present but not displayed)")
        elif arg == 'hf_token' and not value:
             logger.info(f"  {arg}: Not provided")
        else:
            logger.info(f"  {arg}: {value}")

    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.random_seed)
    logger.info(f"Random seeds set to {args.random_seed} for numpy and torch.")

    overall_start_time = time.time()

    try:
        examples_df, test_df = load_datasets(
            args.examples_path, args.test_path, logger, args.test_limit
        )

        model, tokenizer, device = setup_model(
            args.model_name, args.cache_dir, args.gpu_id, args.hf_token, logger
        )

        logger.info(f"Starting batch prediction for {len(test_df)} test comments using {len(examples_df)} few-shot examples.")
        predictions, raw_outputs = batch_predict(
            model, tokenizer, test_df, examples_df, device,
            max_context_length=args.max_length,
            random_seed=args.random_seed,
            checkpoint_interval=args.checkpoint_interval,
            output_dir=args.output_dir,
            logger=logger,
            model_file_prefix=model_file_prefix
        )

        logger.info("Batch prediction completed. Saving results...")
        accuracy, f1 = save_results(
            test_df, predictions, raw_outputs, args.output_dir, logger,
            model_file_prefix=model_file_prefix
        )

        overall_end_time = time.time()
        total_execution_time_secs = overall_end_time - overall_start_time
        total_execution_time_hours = total_execution_time_secs / 3600
        logger.info(f"Total execution time: {total_execution_time_secs:.2f} seconds ({total_execution_time_hours:.2f} hours).")

        logger.info("--- Final Summary ---")
        logger.info(f"Model Used: {args.model_name}")
        logger.info(f"Test Set Size: {len(test_df)} examples")
        logger.info(f"Few-Shot Examples Used: {len(examples_df)}")
        logger.info(f"Achieved Accuracy: {accuracy:.4f}")
        logger.info(f"Achieved F1 Score (for Regional Bias class): {f1:.4f}")
        logger.info(f"Results, logs, and plots saved in: {args.output_dir}")
        logger.info("--- Script Finished Successfully ---")

    except FileNotFoundError as fnf_error:
        logger.error(f"Data file not found: {fnf_error}. Please check paths in arguments or defaults.")
        sys.exit(1)
    except Exception as e:
        logger.error(f"An unexpected error occurred during main execution: {e}", exc_info=True)
        sys.exit(1)
    finally:
        # Clean up CUDA cache if applicable, regardless of success or failure
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        logging.shutdown()


if __name__ == "__main__":
    main()
