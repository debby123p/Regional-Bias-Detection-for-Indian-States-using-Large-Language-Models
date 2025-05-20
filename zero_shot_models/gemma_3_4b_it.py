import os
import pandas as pd
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from tqdm import tqdm
import json
from datetime import datetime
import gc
import argparse
from pathlib import Path
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

"""
Zero-shot binary classification for regional bias detection using Gemma models
with memory optimization techniques.

This script implements a classifier that can identify regional bias in text
using the Gemma language model with a focus on memory efficiency.
"""

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Zero-shot binary classification for regional bias detection using Gemma models')
    
    # Data and output paths
    parser.add_argument('--data-path', type=str, required=True,
                        help='Path to the annotated dataset CSV file')
    parser.add_argument('--output-dir', type=str, default='results',
                        help='Directory to save output files')
    parser.add_argument('--cache-dir', type=str, default=None,
                        help='Directory for model cache')
    
    # Model configuration
    parser.add_argument('--model-name', type=str, 
                        default='google/gemma-3-4b-it',
                        help='Model name or path')
    parser.add_argument('--gpu-id', type=int, default=0,
                        help='GPU ID to use for inference')
    
    # Memory optimization
    parser.add_argument('--quantization', type=str, default='4bit', choices=['4bit', '8bit', 'none'],
                        help='Quantization type (4bit, 8bit, or none)')
    parser.add_argument('--max-split-size-mb', type=int, default=64,
                        help='Maximum CUDA memory split size in MB')
    parser.add_argument('--cpu-offload', action='store_true', 
                        help='Enable CPU offloading for parts of the model')
    
    # Execution parameters
    parser.add_argument('--chunk-size', type=int, default=100,
                        help='Number of comments to process in each chunk')
    parser.add_argument('--max-comment-length', type=int, default=500,
                        help='Maximum length of comment to process')
    parser.add_argument('--max-samples', type=int, default=None,
                        help='Maximum number of samples to process (for testing)')
    
    args = parser.parse_args()
    return args

def setup_environment(args):
    """Set up environment variables for optimal memory usage"""
    # Set GPU device
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)
    
    # Configure memory optimization settings
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = f'expandable_segments:True,max_split_size_mb:{args.max_split_size_mb}'
    
    # Set cache directory if provided
    if args.cache_dir:
        os.environ['TRANSFORMERS_CACHE'] = args.cache_dir
        os.makedirs(args.cache_dir, exist_ok=True)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Optimize GPU settings
    if torch.cuda.is_available():
        print(f"Using GPU {args.gpu_id}: {torch.cuda.get_device_name(0)}")
        # Set additional PyTorch settings for memory optimization
        torch.backends.cudnn.benchmark = False  # Disable for memory savings
        torch.backends.cudnn.deterministic = True  # Enable for deterministic results
    else:
        print("No GPU available. Processing will be extremely slow.")

def free_memory():
    """Free GPU memory to prevent OOM errors"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    gc.collect()

class RegionalBiasClassifier:
    def __init__(self, model_name="google/gemma-3-4b-it", 
                 cache_dir=None, quantization="4bit", cpu_offload=False):
        """
        Initialize the classifier with memory optimizations
        
        Args:
            model_name: Name of the model to load
            cache_dir: Directory for model cache
            quantization: Quantization type (4bit, 8bit, or none)
            cpu_offload: Whether to enable CPU offloading
        """
        print(f"Initializing {model_name} classifier")
        
        # Free memory before loading
        free_memory()
        
        # Load tokenizer 
        tokenizer_kwargs = {
            'trust_remote_code': True,
            'padding_side': "left"
        }
        
        if cache_dir:
            tokenizer_kwargs['cache_dir'] = cache_dir
            
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, **tokenizer_kwargs)
        
        # Set pad token if not available
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        
        # Configure model loading parameters
        model_kwargs = {
            'trust_remote_code': True,
            'low_cpu_mem_usage': True,
            'torch_dtype': torch.float16,
        }
        
        if cache_dir:
            model_kwargs['cache_dir'] = cache_dir
            
        # Configure quantization
        if quantization == "4bit":
            print("Using 4-bit quantization")
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                llm_int8_enable_fp32_cpu_offload=cpu_offload
            )
            model_kwargs['quantization_config'] = bnb_config
        elif quantization == "8bit":
            print("Using 8-bit quantization")
            bnb_config = BitsAndBytesConfig(
                load_in_8bit=True,
                bnb_8bit_use_double_quant=True,
                bnb_8bit_compute_dtype=torch.float16,
                llm_int8_enable_fp32_cpu_offload=cpu_offload
            )
            model_kwargs['quantization_config'] = bnb_config
        else:
            print("Using float16 precision (no quantization)")
        
        # Free memory before model loading
        free_memory()
        
        # Set device mapping for optimal memory usage
        if torch.cuda.is_available():
            model_kwargs['device_map'] = "auto"
        
        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
        
        # Set to evaluation mode
        self.model.eval()
        
        # Free memory again
        free_memory()
        
        if torch.cuda.is_available():
            print(f"Model loaded. Memory allocated: {torch.cuda.memory_allocated(0) / 1e9:.2f} GB")
    
    def classify_with_cot(self, text, max_length=500):
        """
        Classify using Chain-of-Thought prompting
        
        Args:
            text: Text to classify
            max_length: Maximum length of text to process
            
        Returns:
            Dictionary containing classification results
        """
        # Truncate text to save memory
        if len(text) > max_length:
            text = text[:max_length] + "..."
        
        # Gemma-specific prompt format with Chain-of-Thought
        cot_prompt = f"""[INST] You are an expert in identifying regional biases in comments about Indian states and regions. Your task is to determine whether a comment contains regional bias or not. Regional bias includes stereotypes, prejudices, or discriminatory statements about: - Indian states or regions - People from specific Indian states - Cultural, linguistic, economic, political, or infrastructural aspects of Indian regions Please analyze the following comment step by step: Comment: "{text}" Step 1: First, identify if this comment mentions any Indian state, region, or people from specific regions. Step 2: Check if the comment contains any of these elements: - Stereotypical statements about people from a region - Generalizations about a state or its people - Discriminatory language targeting regional identity - Prejudiced views about regional culture, language, or traditions - Biased statements about economic or developmental status - Political stereotypes associated with regions Step 3: Determine if these elements, if present, constitute bias or are merely factual/neutral observations. Step 4: Based on your analysis, classify this comment as: - "regional_bias": If it contains prejudiced, stereotypical, or discriminatory content about Indian regions/states - "non_regional_bias": If it's neutral, factual, or does not contain regional bias Please provide your reasoning followed by your final classification. Format your response as: Reasoning: [Your step-by-step analysis] Classification: [regional_bias/non_regional_bias] [/INST] """
        
        # Free memory before tokenization
        free_memory()
        
        # Tokenize with minimal context
        inputs = self.tokenizer(
            cot_prompt, 
            return_tensors="pt", 
            truncation=True, 
            max_length=768,  # Reduced context size for memory saving
            padding=True
        )
        
        # Move inputs to GPU if available
        if torch.cuda.is_available():
            inputs = {k: v.to("cuda") for k, v in inputs.items()}
        
        with torch.no_grad():
            try:
                # Conservative generation parameters to save memory
                generation_params = {
                    "max_new_tokens": 100,
                    "temperature": 0.1,
                    "do_sample": False,
                    "pad_token_id": self.tokenizer.pad_token_id,
                    "eos_token_id": self.tokenizer.eos_token_id
                }
                
                # Generate response
                outputs = self.model.generate(**inputs, **generation_params)
                    
            except torch.cuda.OutOfMemoryError:
                free_memory()
                
                # Emergency fallback with more restricted parameters
                print("Retrying with more restricted generation parameters")
                generation_params["max_new_tokens"] = 50
                outputs = self.model.generate(**inputs, **generation_params)
        
        # Decode the response
        full_response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract only the generated portion (after the prompt)
        response = full_response[len(self.tokenizer.decode(inputs['input_ids'][0], skip_special_tokens=True)):]
        
        # Clear intermediate tensors
        del outputs, inputs
        free_memory()
        
        # Extract classification and reasoning
        return self._parse_response(response, full_response)
    
    def _parse_response(self, response, full_response):
        """Parse the model's response to extract classification and reasoning"""
        # Default classification
        classification = "non_regional_bias"
        reasoning = response.strip()
        
        # Try to find classification in the response
        lower_response = response.lower()
        
        # Look for explicit classification
        if "classification:" in lower_response:
            parts = lower_response.split("classification:", 1)
            if len(parts) > 1:
                class_text = parts[1].strip()
                class_word = class_text.split()[0] if class_text.split() else ""
                
                if "regional" in class_word and "bias" in class_word and "non" not in class_word:
                    classification = "regional_bias"
                elif any(x in class_word for x in ["non-regional", "non_regional", "nonregional"]):
                    classification = "non_regional_bias"
        
        # If no explicit classification, look for keywords
        if classification == "non_regional_bias" and ("stereotype" in lower_response or "bias" in lower_response or "prejudice" in lower_response):
            if "no stereotype" in lower_response or "no bias" in lower_response or "not biased" in lower_response or "neutral" in lower_response:
                classification = "non_regional_bias"
            else:
                classification = "regional_bias"
        
        # Extract reasoning separately
        if "classification:" in lower_response:
            reasoning = lower_response.split("classification:")[0].strip()
        
        # Check for explicit mentions of regional bias
        if classification == "non_regional_bias" and "regional bias" in lower_response and "non" not in lower_response:
            classification = "regional_bias"
        
        return {
            "classification": classification,
            "reasoning": reasoning,
            "full_response": full_response
        }

def save_results(results, output_dir, timestamp, prefix="results"):
    """Save classification results to CSV and JSON"""
    # Extract predictions
    predictions_data = []
    for result in results:
        predictions_data.append({
            'index': result['index'],
            'comment': result['original_comment'],
            'prediction': result['classification'],
            'prediction_binary': 1 if result['classification'] == 'regional_bias' else 0,
            'reasoning': result.get('reasoning', '')
        })
    
    # Save as CSV
    df_predictions = pd.DataFrame(predictions_data)
    predictions_csv = os.path.join(output_dir, f"{prefix}_predictions_{timestamp}.csv")
    df_predictions.to_csv(predictions_csv, index=False)
    print(f"Predictions saved to CSV: {predictions_csv}")
    
    # Save JSON
    json_file = os.path.join(output_dir, f"{prefix}_results_{timestamp}.json")
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"Results saved to JSON: {json_file}")
    
    return predictions_csv, df_predictions

def process_dataset(classifier, df, max_comment_length=500):
    """Process dataset with a focus on memory efficiency"""
    results = []
    predictions = []
    ground_truth = []
    
    # Process each comment individually
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing comments"):
        comment = str(row['Comment'])
        
        # Free memory before processing
        free_memory()
        
        # Get classification
        result = classifier.classify_with_cot(comment, max_length=max_comment_length)
        
        # Store results
        result['index'] = idx
        result['original_comment'] = comment
        results.append(result)
        
        # Extract prediction
        pred_value = 1 if result['classification'] == 'regional_bias' else 0
        predictions.append(pred_value)
        
        # Extract ground truth if available
        if 'Level-1' in df.columns:
            gt_value = 1 if float(row['Level-1']) > 0 else 0
            ground_truth.append(gt_value)
        elif 'Score' in df.columns:
            gt_value = 1 if float(row['Score']) > 0 else 0
            ground_truth.append(gt_value)
    
    return results, predictions, ground_truth

def generate_evaluation_report(predictions, ground_truth, output_dir, timestamp, model_name):
    """Generate and save evaluation metrics and visualizations"""
    if not ground_truth or len(ground_truth) != len(predictions):
        print("Cannot generate evaluation report: ground truth missing or mismatched")
        return
    
    # Create short model name for file naming
    model_short_name = model_name.split('/')[-1] if '/' in model_name else model_name
    
    # Compute metrics
    report = classification_report(ground_truth, predictions, 
                                  target_names=['Non-Regional Bias', 'Regional Bias'],
                                  output_dict=True)
    report_text = classification_report(ground_truth, predictions, 
                                      target_names=['Non-Regional Bias', 'Regional Bias'])
    conf_matrix = confusion_matrix(ground_truth, predictions)
    
    # Print metrics
    print("\n=== Classification Performance ===")
    print("Classification Report:")
    print(report_text)
    print("Confusion Matrix:")
    print(conf_matrix)
    
    # Save report as text file
    report_file = os.path.join(output_dir, f"classification_report_{timestamp}.txt")
    with open(report_file, 'w') as f:
        f.write(f"=== Classification Report - {model_short_name} ===\n\n")
        f.write(f"Model: {model_name}\n")
        f.write(f"Timestamp: {timestamp}\n")
        f.write(f"Total Samples: {len(predictions)}\n\n")
        f.write(report_text)
        f.write("\n\n=== Confusion Matrix ===\n")
        f.write(str(conf_matrix))
        f.write("\n\nRows: Actual labels\n")
        f.write("Columns: Predicted labels\n")
    
    # Create visualizations directory
    viz_dir = os.path.join(output_dir, "visualizations")
    os.makedirs(viz_dir, exist_ok=True)
    
    # 1. Confusion Matrix visualization
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Non-Regional', 'Regional'],
                yticklabels=['Non-Regional', 'Regional'])
    plt.title(f'Confusion Matrix - {model_short_name}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    confusion_matrix_file = os.path.join(viz_dir, f"confusion_matrix_{timestamp}.png")
    plt.savefig(confusion_matrix_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Class Distribution visualization
    plt.figure(figsize=(10, 6))
    class_counts = [
        sum(label == 0 for label in ground_truth),
        sum(label == 1 for label in ground_truth)
    ]
    pred_counts = [
        sum(pred == 0 for pred in predictions),
        sum(pred == 1 for pred in predictions)
    ]
    
    x = np.arange(2)
    width = 0.35
    
    plt.bar(x - width/2, class_counts, width, label='Actual', color='skyblue')
    plt.bar(x + width/2, pred_counts, width, label='Predicted', color='lightcoral')
    
    plt.xlabel('Class')
    plt.ylabel('Count')
    plt.title('Class Distribution: Actual vs Predicted')
    plt.xticks(x, ['Non-Regional', 'Regional'])
    plt.legend()
    
    # Add count labels on bars
    for i, count in enumerate(class_counts):
        plt.text(i - width/2, count + 5, str(count), ha='center')
    for i, count in enumerate(pred_counts):
        plt.text(i + width/2, count + 5, str(count), ha='center')
    
    distribution_file = os.path.join(viz_dir, f"class_distribution_{timestamp}.png")
    plt.savefig(distribution_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Evaluation report saved: {report_file}")
    print(f"Visualizations saved to: {viz_dir}")

def main():
    """Main execution function with memory optimization"""
    # Parse command line arguments
    args = parse_args()
    
    # Set up environment variables
    setup_environment(args)
    
    # Generate timestamp for this run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Verify input file exists
    data_path = Path(args.data_path)
    if not data_path.exists():
        print(f"Input file not found: {data_path}")
        return
    
    # Load data
    print(f"Loading data from {data_path}")
    df = pd.read_csv(data_path)
    print(f"Data shape: {df.shape}")
    
    # Validate data
    if 'Comment' not in df.columns:
        # Try to find a suitable column
        text_columns = [col for col in df.columns if df[col].dtype == 'object']
        if text_columns:
            print(f"'Comment' column not found. Using '{text_columns[0]}' instead.")
            df['Comment'] = df[text_columns[0]]
        else:
            raise ValueError("No suitable text column found in the data")
    
    # Apply max_samples limit if specified
    if args.max_samples and args.max_samples < len(df):
        print(f"Limiting to {args.max_samples} samples for processing")
        df = df.iloc[:args.max_samples].copy()
    
    # Initialize classifier with memory optimizations
    classifier = RegionalBiasClassifier(
        model_name=args.model_name,
        cache_dir=args.cache_dir,
        quantization=args.quantization,
        cpu_offload=args.cpu_offload
    )
    
    # Process in chunks to avoid memory issues
    total_rows = len(df)
    all_results = []
    all_predictions = []
    all_ground_truth = []
    chunk_size = args.chunk_size
    
    print(f"Processing {total_rows} comments in chunks of {chunk_size}...")
    
    # Process in chunks
    for chunk_start in range(0, total_rows, chunk_size):
        chunk_end = min(chunk_start + chunk_size, total_rows)
        chunk_num = (chunk_start // chunk_size) + 1
        print(f"Processing chunk {chunk_num}: rows {chunk_start} to {chunk_end-1}")
        
        # Process chunk
        chunk_df = df.iloc[chunk_start:chunk_end].copy()
        
        try:
            # Process dataset chunk
            chunk_results, chunk_predictions, chunk_ground_truth = process_dataset(
                classifier,
                chunk_df,
                max_comment_length=args.max_comment_length
            )
            
            # Add to cumulative results
            all_results.extend(chunk_results)
            all_predictions.extend(chunk_predictions)
            if chunk_ground_truth:
                all_ground_truth.extend(chunk_ground_truth)
            
            # Free memory after processing chunk
            free_memory()
            
        except torch.cuda.OutOfMemoryError:
            print(f"CUDA out of memory during chunk processing")
            print(f"Reducing chunk size and retrying for remaining chunks...")
            
            # Reduce chunk size for future chunks
            new_chunk_size = max(10, chunk_size // 2)
            print(f"Reducing chunk size from {chunk_size} to {new_chunk_size}")
            chunk_size = new_chunk_size
            continue
        
        print(f"Successfully processed chunk {chunk_num}")
    
    # Save final results
    final_csv, final_df = save_results(
        all_results, 
        args.output_dir, 
        timestamp
    )
    
    # Generate summary statistics
    regional_bias_count = all_predictions.count(1)
    non_regional_bias_count = all_predictions.count(0)
    
    print("\n=== Classification Summary ===")
    print(f"Total comments classified: {len(all_predictions)}")
    print(f"Regional bias comments: {regional_bias_count} ({regional_bias_count/len(all_predictions)*100:.2f}%)")
    print(f"Non-regional bias comments: {non_regional_bias_count} ({non_regional_bias_count/len(all_predictions)*100:.2f}%)")
    
    print(f"Complete results saved to: {final_csv}")
    
    # Generate evaluation report if ground truth is available
    if all_ground_truth and len(all_ground_truth) == len(all_predictions):
        generate_evaluation_report(
            all_predictions, 
            all_ground_truth, 
            args.output_dir, 
            timestamp, 
            args.model_name
        )

if __name__ == "__main__":
    # Check for GPU and set optimizations
    if torch.cuda.is_available():
        print(f"CUDA is available. Found {torch.cuda.device_count()} GPU(s)")
        
        # Optimize PyTorch for memory efficiency
        torch.backends.cudnn.benchmark = False  # Disable for memory savings
        torch.backends.cudnn.deterministic = True  # Enable for deterministic results
    else:
        print("CUDA is not available. Using CPU only (will be very slow).")
    
    # Run classifier
    print("Starting regional bias classification with Gemma model...")
    main()
