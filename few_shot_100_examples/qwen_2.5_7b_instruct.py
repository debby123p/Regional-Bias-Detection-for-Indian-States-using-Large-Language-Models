import os
# Select to use only GPU 0 (the first GPU)
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # This makes GPU 0 appear as the only device

import pandas as pd
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import time
import logging
from datetime import datetime
from huggingface_hub import login
import re

# Set environment variables to change cache locations BEFORE importing huggingface modules
CACHE_DIR = "/DATA2/akash/venvs/debasmita/model_cache"
os.makedirs(CACHE_DIR, exist_ok=True)
os.environ["TRANSFORMERS_CACHE"] = CACHE_DIR
os.environ["HF_HOME"] = CACHE_DIR
os.environ["HF_DATASETS_CACHE"] = CACHE_DIR
print(f"Set Hugging Face cache directories to: {CACHE_DIR}")

# Set up logging
LOG_DIR = "/DATA2/akash/venvs/debasmita/logs"
os.makedirs(LOG_DIR, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f"{LOG_DIR}/qwen2.5_7b_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Hugging Face authentication
HF_TOKEN = "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"

# Set up directories
EXAMPLE_PATH = "/DATA2/akash/venvs/debasmita/data/50_examples_few_shot_classification_dataset.csv"
TEST_PATH = "/DATA2/akash/venvs/debasmita/data/annotated_experiment_phase_new - Sheet1.csv"
OUTPUT_DIR = "/DATA2/akash/venvs/debasmita/results/few_shot_50"

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Model name for saving files
MODEL_NAME = "qwen2.5_7b_instruct"

# Configure for GPU 0 exclusively
torch.cuda.empty_cache()
torch.cuda.set_device(0)  # Ensure we're using GPU 0
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Set random seed for reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

print(f"Using device: {device}")
print(f"GPU available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU model: {torch.cuda.get_device_name(0)}")
    print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

# Function to clean text
def clean_text(text):
    """Clean and normalize text for model input."""
    text = str(text).lower()
    text = re.sub(r"http\S+", "", text)  # Remove URLs
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)  # Remove special characters
    text = re.sub(r"\s+", " ", text).strip()  # Remove extra spaces
    return text

def load_datasets():
    """Load the example and test datasets"""
    # Load few-shot examples - this contains our examples
    examples_df = pd.read_csv(EXAMPLE_PATH)
    
    # Load full dataset for testing
    test_df = pd.read_csv(TEST_PATH)
    
    # Process the datasets
    logger.info(f"Loaded {len(examples_df)} examples from {EXAMPLE_PATH}")
    logger.info(f"Loaded {len(test_df)} total test comments from {TEST_PATH}")
    
    # Split the examples into bias and non-bias
    bias_examples = examples_df[examples_df['Level-1'] >= 1]
    non_bias_examples = examples_df[examples_df['Level-1'] == 0]
    logger.info(f"Found {len(bias_examples)} bias examples and {len(non_bias_examples)} non-bias examples")
    
    # Ensure there's no overlap between examples and test data
    examples_comments = set(examples_df['Comment'].str.strip())
    test_df = test_df[~test_df['Comment'].str.strip().isin(examples_comments)]
    logger.info(f"After removing overlapping comments, {len(test_df)} test comments remain")
    
    # Clean comments
    logger.info("Cleaning comment text...")
    test_df["Cleaned_Comment"] = test_df["Comment"].apply(clean_text)
    examples_df["Cleaned_Comment"] = examples_df["Comment"].apply(clean_text)
    
    return examples_df, test_df

def create_few_shot_prompt(examples_df, comment):
    """
    Create a prompt for few-shot learning with examples and the target comment.
    Uses all examples from the example dataset.
    
    Args:
        examples_df: DataFrame with example comments
        comment: The comment to classify
        
    Returns:
        Formatted prompt string
    """
    # Split examples by class
    bias_examples = examples_df[examples_df['Level-1'] >= 1]
    non_bias_examples = examples_df[examples_df['Level-1'] == 0]
    
    # Verify we have the expected number of examples
    logger.info(f"Creating prompt with {len(bias_examples)} regional bias and {len(non_bias_examples)} non-regional bias examples")
    
    # Combine all examples and shuffle
    all_examples = examples_df.copy()
    all_examples = all_examples.sample(frac=1, random_state=RANDOM_SEED)  # Shuffle all examples
    
    # Create the prompt - Qwen style
    prompt = "You are an expert in identifying regional biases in comments about Indian states and regions. "
    prompt += "Task: Classify if the comment contains regional bias related to Indian states or regions.\n\n"
    prompt += "Instructions:\n"
    prompt += "- Regional Bias (1): Comments that contain stereotypes, prejudices, or biases about specific Indian states or regions.\n"
    prompt += "- Non-Regional Bias (0): Comments that don't contain regional stereotypes or biases about Indian states.\n\n"
    prompt += "Examples:\n"
    
    for i, row in all_examples.iterrows():
        prompt += f"Comment: \"{row['Cleaned_Comment']}\"\n"
        prompt += f"Classification: {int(row['Level-1'])}\n\n"
    
    prompt += f"Now classify this comment:\n\"{comment}\"\nClassification:"
    
    return prompt

def load_model():
    """Load Qwen model and tokenizer with optimized settings"""
    logger.info("Loading Qwen/Qwen2.5-7B-Instruct model...")
    start_time = time.time()
    
    model_id = "Qwen/Qwen2.5-7B-Instruct"
    hf_token = HF_TOKEN
    
    # Print cache locations to ensure they're correctly set
    logger.info(f"Using cache directory: {CACHE_DIR}")
    logger.info(f"TRANSFORMERS_CACHE: {os.environ['TRANSFORMERS_CACHE']}")
    logger.info(f"HF_HOME: {os.environ['HF_HOME']}")
    
    # Configure to use the maximum available GPU memory
    tokenizer = AutoTokenizer.from_pretrained(
        model_id, 
        token=hf_token,
        cache_dir=CACHE_DIR,
        trust_remote_code=True  # Needed for Qwen models
    )
    
    # For 7B model with 100 examples, use 8-bit quantization
    quantization_config = BitsAndBytesConfig(
        load_in_8bit=True,           # 8-bit quantization
        bnb_8bit_use_double_quant=True,
        bnb_8bit_compute_dtype=torch.float16
    )
    
    # Load with optimized parameters for GPU 0 (40GB A100-PCIE)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        token=hf_token,
        cache_dir=CACHE_DIR,
        quantization_config=quantization_config,
        device_map="cuda:0",  # Explicitly use only GPU 0
        low_cpu_mem_usage=True,
        trust_remote_code=True  # Needed for Qwen models
    )
    
    elapsed_time = time.time() - start_time
    logger.info(f"Model loaded in {elapsed_time:.2f} seconds")
    
    return model, tokenizer

def predict_with_model(model, tokenizer, prompt, max_tokens=10):
    """
    Generate prediction using model.
    
    Args:
        model: The model
        tokenizer: The tokenizer for the model
        prompt: Input prompt text
        max_tokens: Maximum number of tokens to generate
        
    Returns:
        Predicted class (0 or 1)
    """
    try:
        # Tokenize the prompt with truncation to ensure it fits in context window
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=4096).to(device)
        
        # Generate response
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=0.1,  # Low temperature for more deterministic outputs
                do_sample=False,  # Don't sample for deterministic generation
                num_beams=1,      # Simple greedy decoding
                pad_token_id=tokenizer.eos_token_id
            )
        
        # Decode the generated text
        full_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract just the model's response (after our prompt)
        prompt_length = len(prompt)
        if full_output.startswith(prompt):
            generated_text = full_output[prompt_length:].strip()
        else:
            # If we can't find the exact prompt (might be tokenization differences)
            generated_text = full_output[-50:].strip()  # Just get the end
        
        logger.debug(f"Generated text: {generated_text}")
        
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
        logger.warning(f"Could not determine clear classification, using default 0. Response: {last_part}")
        return 0, full_output
        
    except Exception as e:
        logger.error(f"Error in prediction: {e}")
        return 0, f"ERROR: {str(e)}"  # Default to the most common class if there's an error

def batch_predict(model, tokenizer, test_df, examples_df, batch_size=1):
    """Process comments in batches for inference"""
    predictions = []
    raw_outputs = []
    
    # Get test comments
    test_comments = test_df["Cleaned_Comment"].tolist()
    
    # Process examples individually
    for i in range(0, len(test_comments)):
        comment = test_comments[i]
        
        try:
            # Generate prompt with all examples
            prompt = create_few_shot_prompt(examples_df, comment)
            
            # Tokenize to check length
            tokens = tokenizer(prompt, return_tensors="pt", truncation=False)
            input_ids_length = tokens.input_ids.shape[1]
            
            if input_ids_length > 4000:
                logger.warning(f"Prompt is very long ({input_ids_length} tokens). Using truncation.")
                # If too long, truncate to ensure it fits in context window
                prompt = "You are an expert in identifying regional biases in comments about Indian states and regions. "
                prompt += "Task: Classify if the comment contains regional bias related to Indian states or regions.\n\n"
                prompt += "Instructions:\n"
                prompt += "- Regional Bias (1): Comments that contain stereotypes, prejudices, or biases about specific Indian states or regions.\n"
                prompt += "- Non-Regional Bias (0): Comments that don't contain regional stereotypes or biases about Indian states.\n\n"
                prompt += "Examples are provided separately. Based on these instructions:\n\n"
                prompt += f"Classify this comment:\n\"{comment}\"\nClassification:"
            
            # Get prediction
            prediction, raw_output = predict_with_model(model, tokenizer, prompt)
            
            # Store results
            predictions.append(prediction)
            raw_outputs.append(raw_output)
            
            # Print progress
            if (i + 1) % 10 == 0 or i == 0:
                logger.info(f"Processed example {i+1}/{len(test_comments)}")
                logger.info(f"Decision: {prediction} (0=non-regional, 1=regional)")
            
        except Exception as e:
            logger.error(f"Error processing comment {i+1}: {e}")
            predictions.append(0)  # Default to non_regional_bias on error
            raw_outputs.append(f"ERROR: {str(e)}")
        
        # Save checkpoint every 10 examples
        if (i + 1) % 10 == 0 or i == len(test_comments) - 1:
            checkpoint_df = pd.DataFrame({
                'Comment': test_df['Comment'].iloc[:i+1].tolist(),
                'Cleaned_Comment': test_df['Cleaned_Comment'].iloc[:i+1].tolist(),
                'True_Label': test_df['Level-1'].iloc[:i+1].tolist(),
                'Predicted': predictions[:i+1],
                'Model_Output': raw_outputs[:i+1]
            })
            checkpoint_path = os.path.join(OUTPUT_DIR, f"{MODEL_NAME}_checkpoint_{i+1}.csv")
            checkpoint_df.to_csv(checkpoint_path, index=False)
            logger.info(f"Saved checkpoint at {checkpoint_path}")
        
        # Clear cache to prevent OOM errors
        torch.cuda.empty_cache()
    
    return predictions, raw_outputs

def save_results(test_df, predictions, raw_outputs):
    """Save prediction results, classification report, and confusion matrix"""
    
    # Get true labels
    true_labels = test_df['Level-1'].apply(lambda x: 1 if x >= 1 else 0).tolist()
    
    # Save predictions with raw outputs
    results_df = test_df.copy()
    results_df['Predicted'] = predictions
    results_df['Model_Output'] = raw_outputs
    
    # Create output paths
    predictions_path = os.path.join(OUTPUT_DIR, f"{MODEL_NAME}_predictions.csv")
    report_path = os.path.join(OUTPUT_DIR, f"{MODEL_NAME}_report.txt")
    matrix_path = os.path.join(OUTPUT_DIR, f"{MODEL_NAME}_confusion_matrix.png")
    
    # Save predictions CSV
    results_df.to_csv(predictions_path, index=False)
    logger.info(f"Predictions saved to {predictions_path}")
    
    # Generate and save classification report
    report = classification_report(true_labels, predictions)
    with open(report_path, 'w') as f:
        f.write(f"Classification Report for {MODEL_NAME}\n\n")
        f.write(report)
    logger.info(f"Classification report saved to {report_path}")
    
    # Generate and save confusion matrix
    cm = confusion_matrix(true_labels, predictions)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Non-Regional Bias', 'Regional Bias'],
                yticklabels=['Non-Regional Bias', 'Regional Bias'])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(f'Confusion Matrix - {MODEL_NAME}')
    plt.tight_layout()
    plt.savefig(matrix_path)
    logger.info(f"Confusion matrix saved to {matrix_path}")
    
    # Calculate metrics
    accuracy = accuracy_score(true_labels, predictions)
    
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
    class_counts = results_df['Level-1'].value_counts().sort_index()
    plt.bar(['Non-Regional', 'Regional'], class_counts, color=['#1f77b4', '#ff7f0e'])
    plt.title('Test Set Class Distribution')
    plt.ylabel('Number of Samples')
    
    # Add value labels on top of bars
    for i, v in enumerate(class_counts):
        plt.text(i, v + 5, str(v), ha='center')
    
    # Plot prediction distribution
    plt.subplot(2, 2, 3)
    pred_counts = pd.Series(predictions).value_counts().sort_index()
    plt.bar(['Non-Regional', 'Regional'], pred_counts, color=['#2ca02c', '#d62728'])
    plt.title('Model Predictions')
    plt.ylabel('Number of Samples')
    
    # Add value labels on top of bars
    for i, v in enumerate(pred_counts):
        plt.text(i, v + 5, str(v), ha='center')
    
    # Plot accuracy
    plt.subplot(2, 2, 4)
    plt.bar(['Accuracy'], [accuracy], color='#9467bd')
    plt.ylim(0, 1.0)
    plt.title('Model Accuracy')
    plt.text(0, accuracy + 0.05, f"{accuracy:.4f}", ha='center')
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f"{MODEL_NAME}_results_summary.png"))
    logger.info(f"Saved results summary visualization")
    
    return accuracy

def main():
    """Main execution function"""
    # Start timing
    start_time = time.time()
    
    # Login to Hugging Face
    logger.info("Logging in to Hugging Face...")
    login(token=HF_TOKEN)
    
    # Load datasets
    examples_df, test_df = load_datasets()
    
    # Verify we have the expected number of examples (100 total, 50 of each class)
    bias_count = len(examples_df[examples_df['Level-1'] >= 1])
    non_bias_count = len(examples_df[examples_df['Level-1'] == 0])
    logger.info(f"Example set has {bias_count} regional bias and {non_bias_count} non-regional bias examples")
    
    if bias_count != 50 or non_bias_count != 50:
        logger.warning(f"Expected 50 examples of each class, but found {bias_count} bias and {non_bias_count} non-bias examples")
    
    # Load model and tokenizer
    model, tokenizer = load_model()
    
    # For quicker testing/development, can use a subset
    # Uncomment to test with a smaller number
    # test_df = test_df.head(100)
    
    # Predict using our examples dataset with all 100 examples
    logger.info(f"Processing {len(test_df)} comments with all {len(examples_df)} few-shot examples...")
    predictions, raw_outputs = batch_predict(
        model, tokenizer, test_df, examples_df
    )
    
    # Save results
    accuracy = save_results(test_df, predictions, raw_outputs)
    logger.info(f"Accuracy: {accuracy:.4f}")
    
    # End timing
    end_time = time.time()
    elapsed_hours = (end_time - start_time) / 3600
    logger.info(f"Total execution time: {elapsed_hours:.2f} hours")

if __name__ == "__main__":
    main()