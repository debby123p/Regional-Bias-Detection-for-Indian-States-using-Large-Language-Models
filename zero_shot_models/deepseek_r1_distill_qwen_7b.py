#!/usr/bin/env python3
"""
Zero-shot binary classification for regional bias detection using DeepSeek-R1-Distill-Qwen-7B
with extreme memory optimizations for 8GB GPU (RTX 3070)
Author: Debasmita (Modified)
"""

import os
import pandas as pd
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from tqdm import tqdm
import json
from datetime import datetime
import gc
import sys
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import glob
import argparse

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Set critical environment variables for memory optimization
os.environ['HF_HOME'] = '/home/debasmita/bs_thesis/huggingface_cache'
os.environ['TRANSFORMERS_CACHE'] = '/home/debasmita/bs_thesis/huggingface_cache'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # Use GPU 0
os.environ['TORCH_USE_CUDA_DSA'] = '1'  # Enable device-side assertions
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'  # Better error messages
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True,max_split_size_mb:64'  # Critical for memory fragmentation

# Create cache directory if it doesn't exist
os.makedirs('/home/debasmita/bs_thesis/huggingface_cache', exist_ok=True)

def free_memory():
    """Aggressively free GPU memory"""
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    gc.collect()

class RegionalBiasClassifier:
    def __init__(self, model_name="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"):
        logger.info(f"Initializing {model_name} with extreme memory optimizations")
        logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"Total GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        
        # Aggressively free memory before loading
        free_memory()
        
        try:
            # Load model in OFFLINE mode to avoid loading both tokenizers simultaneously
            # First step: Load tokenizer only
            logger.info("Loading tokenizer only first...")
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                cache_dir="/home/debasmita/bs_thesis/huggingface_cache",
                trust_remote_code=True,
                padding_side="left",
                local_files_only=False  # Allow downloading
            )
            
            # Set pad token if not available
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
            
            # Configure extremely aggressive quantization for 8GB GPU
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,  # 4-bit quantization is essential
                bnb_4bit_use_double_quant=True,  # Use double quantization to further compress
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                llm_int8_enable_fp32_cpu_offload=True  # Enable CPU offloading
            )
            
            # Second step: Free memory before model loading
            free_memory()
            
            # Load model with extremely conservative settings
            logger.info("Loading model with extreme memory optimizations...")
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                quantization_config=bnb_config,
                device_map="auto",  # Let the library decide the optimal device mapping
                cache_dir="/home/debasmita/bs_thesis/huggingface_cache",
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True,
                trust_remote_code=True,
                offload_folder="/home/debasmita/bs_thesis/offload",  # Offload to disk if needed
                offload_state_dict=True,  # Offload state dict to CPU
                local_files_only=False  # Allow downloading
            )
            
            # Set to evaluation mode
            self.model.eval()
            
            # Free memory again
            free_memory()
            
            logger.info(f"Model loaded. Memory allocated: {torch.cuda.memory_allocated(0) / 1e9:.2f} GB")
            logger.info(f"Memory reserved: {torch.cuda.memory_reserved(0) / 1e9:.2f} GB")
            
        except Exception as e:
            logger.error(f"Error initializing model: {e}")
            raise
    
    def classify_with_cot(self, text, max_length=500):
        """
        Classify using Chain-of-Thought prompting with very limited context size
        """
        # Heavily truncate text to save memory
        if len(text) > max_length:
            text = text[:max_length] + "..."
        
        # DeepSeek models respond well to structured prompts - using minimal context
        cot_prompt = f"""<|im_start|>user
Determine if this comment contains regional bias about Indian states or regions:

"{text}"

Regional bias means stereotypes, prejudices, or discrimination about:
- Indian states/regions
- People from specific Indian states
- Cultural, linguistic, economic aspects of Indian regions

First identify any Indian place mentioned. 
Check for stereotypes, generalizations, prejudice, or discriminatory language.
Determine if these are bias or just neutral observations.

Classify as either "regional_bias" or "non_regional_bias".

Format: 
Reasoning: [brief analysis]
Classification: [regional_bias/non_regional_bias]
<|im_end|>
<|im_start|>assistant
"""
        
        try:
            # Free memory before tokenization
            free_memory()
            
            # Tokenize with minimal context
            inputs = self.tokenizer(
                cot_prompt, 
                return_tensors="pt", 
                truncation=True, 
                max_length=768,  # Reduced context size
                padding=True
            ).to("cuda")
            
            # Free memory again
            free_memory()
            
            with torch.no_grad():
                try:
                    # Ultra-conservative generation parameters
                    generation_params = {
                        "max_new_tokens": 100,  # Severely limited output
                        "temperature": 0.1,
                        "do_sample": False,  # Use greedy decoding to save memory
                        "pad_token_id": self.tokenizer.pad_token_id,
                        "eos_token_id": self.tokenizer.eos_token_id,
                        "repetition_penalty": 1.0
                    }
                    
                    # Try generating with very conservative settings
                    outputs = self.model.generate(**inputs, **generation_params)
                        
                except torch.cuda.OutOfMemoryError as e:
                    logger.error(f"CUDA out of memory: {e}")
                    free_memory()
                    
                    # Emergency fallback mode - even more restricted
                    generation_params["max_new_tokens"] = 50
                    outputs = self.model.generate(**inputs, **generation_params)
                    
                except Exception as e:
                    logger.error(f"Error in model generation: {e}")
                    return {
                        "classification": "error",
                        "reasoning": f"Error: {str(e)}",
                        "full_response": ""
                    }
            
            # Decode the response
            response = self.tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
            
            # Clear intermediate tensors
            del outputs, inputs
            free_memory()
            
            # Extract classification and reasoning
            return self._parse_response(response)
            
        except Exception as e:
            logger.error(f"Error in classification: {e}")
            return {
                "classification": "error",
                "reasoning": f"Error: {str(e)}",
                "full_response": ""
            }
    
    def _parse_response(self, response):
        """Parse the model's response to extract classification and reasoning"""
        try:
            lines = response.strip().split('\n')
            reasoning = ""
            classification = ""
            
            for line in lines:
                line = line.strip()
                if line.startswith("Reasoning:"):
                    reasoning = line.replace("Reasoning:", "").strip()
                elif line.startswith("Classification:"):
                    classification = line.replace("Classification:", "").strip().lower()
                    # Clean up the classification
                    if "regional_bias" in classification:
                        if "non" in classification or "not" in classification:
                            classification = "non_regional_bias"
                        else:
                            classification = "regional_bias"
            
            # If reasoning is not found in expected format, try to extract it from full response
            if not reasoning and len(lines) > 0:
                # Try to find text before any "Classification:" line
                classification_idx = next((i for i, line in enumerate(lines) if "Classification:" in line), -1)
                if classification_idx > 0:
                    reasoning = " ".join(lines[:classification_idx])
            
            # Validate classification - make sure we have something usable
            if classification not in ["regional_bias", "non_regional_bias"]:
                logger.warning(f"Invalid classification: {classification}")
                if "bias" in response.lower() and "regional" in response.lower() and "non" not in response.lower():
                    classification = "regional_bias"
                else:
                    classification = "non_regional_bias"
            
            return {
                "classification": classification,
                "reasoning": reasoning,
                "full_response": response
            }
            
        except Exception as e:
            logger.error(f"Error parsing response: {e}")
            return {
                "classification": "error",
                "reasoning": "Error in parsing",
                "full_response": response
            }

def save_results(results, output_dir, timestamp, prefix="deepseek"):
    """Save classification results safely"""
    try:
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Extract predictions
        predictions_data = []
        for result in results:
            predictions_data.append({
                'index': result['index'],
                'comment': result['original_comment'],
                'prediction': result['classification'],
                'prediction_binary': 1 if result['classification'] == 'regional_bias' else 0 if result['classification'] == 'non_regional_bias' else -1,
                'reasoning': result.get('reasoning', '')
            })
        
        # Save as CSV first (most robust)
        df_predictions = pd.DataFrame(predictions_data)
        predictions_csv = os.path.join(output_dir, f"{prefix}_predictions_{timestamp}.csv")
        df_predictions.to_csv(predictions_csv, index=False)
        logger.info(f"Predictions saved to CSV: {predictions_csv}")
        
        # Try to save JSON
        try:
            json_file = os.path.join(output_dir, f"{prefix}_results_{timestamp}.json")
            with open(json_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            logger.info(f"Results saved to JSON: {json_file}")
        except Exception as e:
            logger.error(f"Error saving to JSON: {e}")
        
        return predictions_csv, df_predictions
        
    except Exception as e:
        logger.error(f"Error saving results: {e}")
        return None, None

def process_one_comment(classifier, comment, idx):
    """Process a single comment with maximum error prevention"""
    try:
        # Aggressively free memory before processing
        free_memory()
        
        # Get classification
        result = classifier.classify_with_cot(comment)
        
        # Store results
        result['index'] = idx
        result['original_comment'] = comment
        
        # Extract prediction value
        pred_value = None
        if result['classification'] != 'error':
            pred_value = 1 if result['classification'] == 'regional_bias' else 0
        
        # Free memory again
        free_memory()
        
        return result, pred_value
        
    except Exception as e:
        logger.error(f"Error processing comment {idx}: {e}")
        return {
            "index": idx,
            "original_comment": comment,
            "classification": "error",
            "reasoning": f"Error: {str(e)}",
            "full_response": ""
        }, None

def process_dataset(classifier, df, save_interval=1):
    """Process dataset one comment at a time with frequent saving"""
    results = []
    predictions = []
    ground_truth = []
    
    # Create timestamp for saving
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Output directory for intermediary results
    output_dir = "/home/debasmita/project/new_results/intermediary"
    os.makedirs(output_dir, exist_ok=True)
    
    # Process each comment individually to minimize memory usage
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing comments"):
        comment = str(row['Comment'])
        
        # Process comment
        result, pred_value = process_one_comment(classifier, comment, idx)
        
        # Add to results
        results.append(result)
        if pred_value is not None:
            predictions.append(pred_value)
        
        # Extract ground truth
        if 'Level-1' in df.columns:
            try:
                gt_value = 1 if float(row['Level-1']) > 0 else 0
                ground_truth.append(gt_value)
            except (ValueError, TypeError):
                logger.warning(f"Invalid Level-1 value at index {idx}")
        elif 'Score' in df.columns:
            try:
                gt_value = 1 if float(row['Score']) > 0 else 0
                ground_truth.append(gt_value)
            except (ValueError, TypeError):
                logger.warning(f"Invalid Score value at index {idx}")
        
        # Save results after every comment or interval
        if (idx + 1) % save_interval == 0 or idx == len(df) - 1:
            # Save intermediary results
            intermediary_file = os.path.join(
                output_dir, 
                f"deepseek_comment_{idx+1}_of_{len(df)}_{timestamp}.csv"
            )
            
            # Create DataFrame with current results
            df_intermediary = pd.DataFrame([
                {
                    'index': r['index'],
                    'comment': r['original_comment'],
                    'prediction': r['classification'],
                    'prediction_binary': 1 if r['classification'] == 'regional_bias' else 0 if r['classification'] == 'non_regional_bias' else -1,
                    'reasoning': r.get('reasoning', '')
                }
                for r in results
            ])
            
            # Save to CSV
            df_intermediary.to_csv(intermediary_file, index=False)
            logger.info(f"Saved progress after comment {idx+1} to {intermediary_file}")
        
        # Free memory after each comment
        free_memory()
    
    return results, predictions, ground_truth

def find_latest_checkpoint(output_dir):
    """Find the latest checkpoint to resume from"""
    checkpoint_files = glob.glob(os.path.join(output_dir, "chunks", "*_chunk_*.csv"))
    if not checkpoint_files:
        return None, 0
    
    # Extract chunk numbers
    chunk_nums = []
    for f in checkpoint_files:
        try:
            chunk_num = int(f.split("_chunk_")[-1].split(".")[0])
            chunk_nums.append((chunk_num, f))
        except:
            continue
    
    if not chunk_nums:
        return None, 0
    
    # Find the latest chunk
    latest_chunk_num, latest_file = max(chunk_nums, key=lambda x: x[0])
    logger.info(f"Found checkpoint file: {latest_file} (chunk {latest_chunk_num})")
    
    # Load processed indices
    df_checkpoint = pd.read_csv(latest_file)
    processed_indices = set(df_checkpoint['index'].astype(int).tolist())
    
    return processed_indices, latest_chunk_num

def main():
    """Main function to process all comments with memory management"""
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Process all comments for regional bias detection')
    parser.add_argument('--data-path', default="/home/debasmita/bs_thesis/code/annotated_experiment - Sheet1.csv", 
                        help='Path to the data file')
    parser.add_argument('--output-dir', default="/home/debasmita/project/new_results", 
                        help='Output directory for results')
    parser.add_argument('--chunk-size', type=int, default=100, 
                        help='Number of comments to process in each chunk')
    parser.add_argument('--save-interval', type=int, default=50, 
                        help='Interval to save results during processing')
    parser.add_argument('--resume', action='store_true', 
                        help='Resume from the last checkpoint')
    args = parser.parse_args()
    
    # Configuration
    data_path = args.data_path
    output_dir = args.output_dir
    chunk_size = args.chunk_size
    save_interval = args.save_interval
    
    # Create output directories
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "intermediary"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "chunks"), exist_ok=True)
    
    # Verify input file exists
    if not os.path.exists(data_path):
        logger.error(f"Input file not found: {data_path}")
        logger.info("Checking for alternative paths...")
        
        # Try alternative locations
        alt_paths = [
            "/home/debasmita/bs_thesis/annotated_experiment - Sheet1.csv",
            "/home/debasmita/project/annotated_experiment - Sheet1.csv",
            "/home/debasmita/project/data/annotated_experiment - Sheet1.csv",
            "/home/debasmita/bs_thesis/change.csv"  # Added this alternative path
        ]
        
        for path in alt_paths:
            if os.path.exists(path):
                data_path = path
                logger.info(f"Found input file at alternative location: {data_path}")
                break
        
        if not os.path.exists(data_path):
            logger.error("Could not find input file at any expected location. Exiting.")
            return
    
    try:
        # Load data
        logger.info(f"Loading data from {data_path}")
        df = pd.read_csv(data_path)
        logger.info(f"Data shape: {df.shape}")
        logger.info(f"Columns: {df.columns.tolist()}")
        
        # Validate data
        if 'Comment' not in df.columns:
            # Try to find a suitable column
            text_columns = [col for col in df.columns if df[col].dtype == 'object']
            if text_columns:
                logger.warning(f"'Comment' column not found. Using '{text_columns[0]}' instead.")
                df['Comment'] = df[text_columns[0]]
            else:
                raise ValueError("No suitable text column found in the data")
        
        # Check for resume
        processed_indices = None
        last_chunk_num = 0
        if args.resume:
            processed_indices, last_chunk_num = find_latest_checkpoint(output_dir)
            if processed_indices:
                logger.info(f"Resuming from chunk {last_chunk_num}, {len(processed_indices)} comments already processed")
                # Filter out already processed rows
                df = df[~df.index.isin(processed_indices)].copy()
                logger.info(f"Remaining comments to process: {len(df)}")
        
        # Initialize classifier with DeepSeek model
        try:
            classifier = RegionalBiasClassifier()
        except torch.cuda.OutOfMemoryError as e:
            logger.error(f"CUDA out of memory during model initialization: {e}")
            logger.error("Your GPU (8GB RTX 3070) doesn't have enough memory for this model.")
            logger.error("Please consider using a smaller model or a GPU with more memory.")
            return
        
        # Generate timestamp for output files
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Process chunks of data to avoid memory issues
        total_rows = len(df)
        all_results = []
        all_predictions = []
        all_ground_truth = []
        
        logger.info(f"Processing all {total_rows} comments in chunks of {chunk_size}...")
        
        # Process in chunks
        for chunk_start in range(0, total_rows, chunk_size):
            chunk_end = min(chunk_start + chunk_size, total_rows)
            chunk_num = last_chunk_num + (chunk_start // chunk_size) + 1
            logger.info(f"Processing chunk {chunk_num}: rows {chunk_start} to {chunk_end-1}")
            
            # Process chunk
            chunk_df = df.iloc[chunk_start:chunk_end].copy()
            
            try:
                chunk_results, chunk_predictions, chunk_ground_truth = process_dataset(
                    classifier, chunk_df, save_interval=save_interval
                )
                
                # Add to cumulative results
                all_results.extend(chunk_results)
                if chunk_predictions:
                    all_predictions.extend(chunk_predictions)
                if chunk_ground_truth:
                    all_ground_truth.extend(chunk_ground_truth)
                
                # Save intermediate results after each chunk
                chunk_output_dir = os.path.join(output_dir, "chunks")
                os.makedirs(chunk_output_dir, exist_ok=True)
                save_results(
                    chunk_results, 
                    chunk_output_dir, 
                    f"{timestamp}_chunk_{chunk_num}"
                )
                
                # Free memory after processing chunk
                free_memory()
                
            except torch.cuda.OutOfMemoryError as e:
                logger.error(f"CUDA out of memory during chunk processing: {e}")
                logger.warning(f"Reducing chunk size and retrying for remaining chunks...")
                
                # Save what we have so far before adjusting
                save_results(all_results, output_dir, f"{timestamp}_partial_{chunk_start}")
                
                # Reduce chunk size for future chunks
                new_chunk_size = max(10, chunk_size // 2)
                logger.info(f"Reducing chunk size from {chunk_size} to {new_chunk_size}")
                
                # Process remaining data with smaller chunk size
                for smaller_chunk_start in range(chunk_start, total_rows, new_chunk_size):
                    smaller_chunk_end = min(smaller_chunk_start + new_chunk_size, total_rows)
                    smaller_chunk_num = chunk_num + ((smaller_chunk_start - chunk_start) // new_chunk_size) + 1
                    
                    logger.info(f"Processing smaller chunk {smaller_chunk_num}: rows {smaller_chunk_start} to {smaller_chunk_end-1}")
                    
                    # Process smaller chunk
                    smaller_chunk_df = df.iloc[smaller_chunk_start:smaller_chunk_end].copy()
                    
                    try:
                        smaller_chunk_results, smaller_chunk_predictions, smaller_chunk_ground_truth = process_dataset(
                            classifier, smaller_chunk_df, save_interval=save_interval
                        )
                        
                        # Add to cumulative results
                        all_results.extend(smaller_chunk_results)
                        if smaller_chunk_predictions:
                            all_predictions.extend(smaller_chunk_predictions)
                        if smaller_chunk_ground_truth:
                            all_ground_truth.extend(smaller_chunk_ground_truth)
                        
                        # Save intermediate results after each smaller chunk
                        save_results(
                            smaller_chunk_results, 
                            chunk_output_dir, 
                            f"{timestamp}_smaller_chunk_{smaller_chunk_num}"
                        )
                        
                        # Free memory
                        free_memory()
                        
                    except Exception as e:
                        logger.error(f"Error processing smaller chunk {smaller_chunk_num}: {e}")
                        # Save what we have so far
                        save_results(all_results, output_dir, f"{timestamp}_partial_smaller_{smaller_chunk_start}")
                        continue
                
                # After processing with smaller chunks, break out of the original loop
                break
                
            except Exception as e:
                logger.error(f"Error processing chunk {chunk_num}: {e}")
                # Save what we have so far
                save_results(all_results, output_dir, f"{timestamp}_partial_{chunk_start}")
                continue
            
            # Success message for chunk
            logger.info(f"Successfully processed chunk {chunk_num}")
        
        # Save final complete results
        final_csv, final_df = save_results(all_results, output_dir, timestamp)
        
        # Generate summary statistics
        if all_predictions:
            regional_bias_count = all_predictions.count(1)
            non_regional_bias_count = all_predictions.count(0)
            
            logger.info("\n=== Classification Summary ===")
            logger.info(f"Total comments classified: {len(all_predictions)}")
            logger.info(f"Regional bias comments: {regional_bias_count} ({regional_bias_count/len(all_predictions)*100:.2f}%)")
            logger.info(f"Non-regional bias comments: {non_regional_bias_count} ({non_regional_bias_count/len(all_predictions)*100:.2f}%)")
        
        logger.info(f"Complete results saved to: {final_csv}")
        
        # Evaluate performance if ground truth is available
        if all_ground_truth and len(all_ground_truth) == len(all_predictions):
            try:
                # Compute metrics
                logger.info("\n=== Classification Performance ===")
                report = classification_report(all_ground_truth, all_predictions)
                conf_matrix = confusion_matrix(all_ground_truth, all_predictions)
                
                logger.info("Classification Report:")
                logger.info(report)
                
                logger.info("Confusion Matrix:")
                logger.info(conf_matrix)
                
                # Save report and confusion matrix
                with open(os.path.join(output_dir, f"{timestamp}_metrics.txt"), 'w') as f:
                    f.write("Classification Report:\n")
                    f.write(report)
                    f.write("\nConfusion Matrix:\n")
                    f.write(str(conf_matrix))
                
                # Plot and save confusion matrix
                try:
                    plt.figure(figsize=(8, 6))
                    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                                xticklabels=['Non-Regional Bias', 'Regional Bias'],
                                yticklabels=['Non-Regional Bias', 'Regional Bias'])
                    plt.xlabel('Predicted')
                    plt.ylabel('True')
                    plt.title('Confusion Matrix')
                    plt.tight_layout()
                    plt.savefig(os.path.join(output_dir, f"{timestamp}_confusion_matrix.png"))
                    plt.close()
                except Exception as e:
                    logger.error(f"Error plotting confusion matrix: {e}")
            
            except Exception as e:
                logger.error(f"Error calculating metrics: {e}")
        
    except Exception as e:
        logger.error(f"Error in main execution: {e}")
        raise

if __name__ == "__main__":
    # Check GPU availability
    if not torch.cuda.is_available():
        logger.error("No GPU available!")
        exit(1)
    
    logger.info(f"Python version: {sys.version}")
    logger.info(f"PyTorch version: {torch.__version__}")
    logger.info(f"CUDA version: {torch.version.cuda}")
    logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
    logger.info(f"CUDA device-side assertions enabled: {os.environ.get('TORCH_USE_CUDA_DSA', '0')}")
    
    # Optimize GPU usage for 8GB GPU
    torch.backends.cudnn.benchmark = False  # Disable for memory savings
    torch.backends.cudnn.deterministic = True  # Enable for memory savings
    
    # Run classification
    logger.info("Starting classification with DeepSeek model and extreme memory optimizations...")
    main()
