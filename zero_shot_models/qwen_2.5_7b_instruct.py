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
import logging
import glob
import argparse
from pathlib import Path
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Zero-shot binary classification for regional bias detection using LLMs like Qwen with memory optimization')
    
    # Data and output paths
    parser.add_argument('--data-path', type=str, required=True,
                        help='Path to the annotated dataset CSV file')
    parser.add_argument('--output-dir', type=str, default='results/qwen/',
                        help='Directory to save output files')
    parser.add_argument('--cache-dir', type=str, default=None,
                        help='Directory for model cache')
    parser.add_argument('--offload-dir', type=str, default=None,
                        help='Directory for model offloading')
    
    # Model configuration
    parser.add_argument('--model-name', type=str, 
                        default='Qwen/Qwen2.5-7B-Instruct', # Changed default model
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
    parser.add_argument('--save-interval', type=int, default=50,
                        help='Interval to save results during processing (within a chunk)')
    parser.add_argument('--max-comment-length', type=int, default=500,
                        help='Maximum length of comment to process (will be truncated if longer)')
    parser.add_argument('--max-samples', type=int, default=None,
                        help='Maximum number of samples to process (for testing)')
    
    # Control flow
    parser.add_argument('--resume', action='store_true',
                        help='Resume from the last checkpoint')
    parser.add_argument('--debug', action='store_true',
                        help='Enable debug mode with additional logging')
    
    args = parser.parse_args()
    return args

def get_model_slug(model_name_or_path):
    """Generate a filesystem-friendly slug from the model name."""
    return Path(model_name_or_path).name.lower().replace('-', '_').replace('.', '_')

def setup_environment(args):
    """Set up environment variables and directories for optimal memory usage"""
    # Set GPU device
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)
    
    # Configure memory optimization settings
    os.environ['TORCH_USE_CUDA_DSA'] = '1'  # Enable device-side assertions
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'  # Better error messages
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = f'expandable_segments:True,max_split_size_mb:{args.max_split_size_mb}'
    
    # Set cache directory if provided
    if args.cache_dir:
        os.environ['HF_HOME'] = args.cache_dir
        os.environ['TRANSFORMERS_CACHE'] = args.cache_dir
        os.makedirs(args.cache_dir, exist_ok=True)
        logger.info(f"Using cache directory: {args.cache_dir}")
    
    # Set up offload directory if provided
    if args.offload_dir:
        os.makedirs(args.offload_dir, exist_ok=True)
        logger.info(f"Using offload directory: {args.offload_dir}")
    
    # Create output directories
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "intermediary"), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "chunks"), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "visualizations"), exist_ok=True)
    
    # Set up file logger
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_slug = get_model_slug(args.model_name)
    log_file = os.path.join(args.output_dir, f"{model_slug}_classification_{timestamp}.log")
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(file_handler)
    
    # Set log level
    if args.debug:
        logger.setLevel(logging.DEBUG)
        logger.debug("Debug mode enabled")
    
    # Log system information
    logger.info(f"Python version: {sys.version}")
    logger.info(f"PyTorch version: {torch.__version__}")
    logger.info(f"CUDA version: {torch.version.cuda if torch.cuda.is_available() else 'N/A'}")
    
    # Optimize GPU settings
    if torch.cuda.is_available():
        # After CUDA_VISIBLE_DEVICES is set, device 0 is the target GPU
        logger.info(f"Using GPU {args.gpu_id} (mapped to cuda:0): {torch.cuda.get_device_name(0)}")
        logger.info(f"Total GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    else:
        logger.warning("No GPU available! Processing will be extremely slow.")

def free_memory():
    """Aggressively free GPU memory"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    gc.collect()

class RegionalBiasClassifier:
    def __init__(self, model_name="Qwen/Qwen2.5-7B-Instruct", # Changed default model
                 cache_dir=None, offload_dir=None, quantization="4bit", cpu_offload=False):
        """
        Initialize the classifier with extreme memory optimizations
        
        Args:
            model_name: Name of the model to load
            cache_dir: Directory for model cache
            offload_dir: Directory for model offloading
            quantization: Quantization type (4bit, 8bit, or none)
            cpu_offload: Whether to enable CPU offloading
        """
        self.model_name = model_name # Store model_name
        logger.info(f"Initializing {model_name} with extreme memory optimizations")
        if torch.cuda.is_available():
            logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}") # Assumes CUDA_VISIBLE_DEVICES set
            logger.info(f"Total GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        
        free_memory()
        
        try:
            logger.info("Loading tokenizer...")
            tokenizer_kwargs = {
                'trust_remote_code': True,
                'padding_side': "left",
            }
            if cache_dir:
                tokenizer_kwargs['cache_dir'] = cache_dir
                
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, **tokenizer_kwargs)
            
            if self.tokenizer.pad_token is None:
                logger.warning("Tokenizer does not have a pad_token. Setting pad_token to eos_token.")
                self.tokenizer.pad_token = self.tokenizer.eos_token
            # Ensure pad_token_id is also set if pad_token was manually assigned
            if self.tokenizer.pad_token_id is None and self.tokenizer.pad_token is not None:
                 self.tokenizer.pad_token_id = self.tokenizer.convert_tokens_to_ids(self.tokenizer.pad_token)


            model_kwargs = {
                'trust_remote_code': True,
                'low_cpu_mem_usage': True,
                'torch_dtype': torch.float16, # Qwen2 might prefer bfloat16 if available, but float16 is safer
            }
            
            if cache_dir:
                model_kwargs['cache_dir'] = cache_dir
                
            if offload_dir:
                model_kwargs['offload_folder'] = offload_dir
                model_kwargs['offload_state_dict'] = True # For accelerate aenable_cpu_offload
            
            if quantization == "4bit":
                logger.info("Using 4-bit quantization for model loading")
                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.float16,
                )
                if cpu_offload: # CPU offloading for 4-bit is typically handled by device_map + accelerate
                    logger.info("CPU offload for 4-bit is handled via device_map='auto'. Ensure 'accelerate' is installed.")
                model_kwargs['quantization_config'] = bnb_config
            elif quantization == "8bit":
                logger.info("Using 8-bit quantization for model loading")
                bnb_config = BitsAndBytesConfig(
                    load_in_8bit=True,
                    bnb_8bit_use_double_quant=True, # Might not be available/standard for 8bit
                    bnb_8bit_compute_dtype=torch.float16, # Check if Qwen needs specific compute dtype
                    llm_int8_enable_fp32_cpu_offload=cpu_offload
                )
                model_kwargs['quantization_config'] = bnb_config
            else:
                logger.info("Using float16 precision (no quantization)")
            
            free_memory()
            logger.info("Loading model with memory optimizations...")
            
            if torch.cuda.is_available():
                logger.info("Using 'auto' device mapping for optimal memory usage")
                model_kwargs['device_map'] = "auto" # Handles CPU offload with accelerate
            
            self.model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
            self.model.eval()
            free_memory()
            
            if torch.cuda.is_available():
                logger.info(f"Model loaded. Memory allocated: {torch.cuda.memory_allocated(0) / 1e9:.2f} GB")
                logger.info(f"Memory reserved: {torch.cuda.memory_reserved(0) / 1e9:.2f} GB")
            
        except Exception as e:
            logger.error(f"Error initializing model: {e}")
            raise
    
    def classify_with_cot(self, text, max_length=500):
        if len(text) > max_length:
            text = text[:max_length] + "..."
        
        # Qwen models use ChatML format.
        # The tokenizer can apply the template, or we can construct it manually.
        # Manual construction similar to DeepSeek's prompt structure:
        messages = [
            # Optional: Add a system prompt if desired for more specific instructions,
            # but often the user prompt is sufficient for this task.
            # {"role": "system", "content": "You are an AI assistant that classifies text for regional bias."},
            {"role": "user", "content": f"""Determine if this comment contains regional bias about Indian states or regions:

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
Classification: [regional_bias/non_regional_bias]"""}
        ]
        
        # Use tokenizer's chat template for correct formatting
        try:
            # `add_generation_prompt=True` adds the assistant's turn starter (e.g., "<|im_start|>assistant\n")
            prompt = self.tokenizer.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=True 
            )
        except Exception as e:
            logger.error(f"Error applying chat template: {e}. Falling back to manual prompt.")
            # Fallback to manual prompt similar to the original script, adjust if Qwen needs specific system prompt
            prompt = f"""<|im_start|>user
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
            free_memory()
            inputs = self.tokenizer(
                prompt, 
                return_tensors="pt", 
                truncation=True, 
                max_length=768, # Max input tokens
                padding=True
            )
            
            if torch.cuda.is_available():
                inputs = {k: v.to("cuda") for k, v in inputs.items()}
            
            free_memory()
            
            with torch.no_grad():
                generation_params = {
                    "max_new_tokens": 100,
                    "temperature": 0.1,
                    "do_sample": False,
                    "pad_token_id": self.tokenizer.pad_token_id,
                    "eos_token_id": self.tokenizer.eos_token_id, # Qwen might have multiple, ensure this is the primary one
                    "repetition_penalty": 1.0
                }
                
                try:
                    outputs = self.model.generate(**inputs, **generation_params)
                except torch.cuda.OutOfMemoryError as e_oom:
                    logger.error(f"CUDA out of memory during generation: {e_oom}")
                    free_memory()
                    logger.info("Retrying with more restricted generation parameters (max_new_tokens=50)")
                    generation_params["max_new_tokens"] = 50
                    outputs = self.model.generate(**inputs, **generation_params)
            
            response_ids = outputs[0][inputs['input_ids'].shape[1]:]
            response = self.tokenizer.decode(response_ids, skip_special_tokens=True)
            
            del outputs, inputs, response_ids
            free_memory()
            
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
        # This parsing logic is dependent on the model following instructions.
        # It's kept similar to the original for consistency.
        try:
            lines = response.strip().split('\n')
            reasoning = ""
            classification = ""
            
            for line in lines:
                line_l = line.lower().strip()
                if line_l.startswith("reasoning:"):
                    reasoning = line[len("Reasoning:"):].strip()
                elif line_l.startswith("classification:"):
                    classification_text = line[len("Classification:"):].strip().lower()
                    if "non_regional_bias" in classification_text or "non regional bias" in classification_text:
                        classification = "non_regional_bias"
                    elif "regional_bias" in classification_text or "regional bias" in classification_text:
                        classification = "regional_bias"
            
            if not reasoning and len(lines) > 0:
                class_idx = -1
                for i, line_content in enumerate(lines):
                    if "classification:" in line_content.lower():
                        class_idx = i
                        break
                if class_idx > 0 :
                    reasoning = " ".join(lines[:class_idx]).strip()
                elif class_idx == -1 and classification: # Only classification found
                    reasoning = "N/A (classification found, reasoning not explicitly)"
                else: # No classification keyword, take the whole response as reasoning if short
                    reasoning = response if len(response) < 200 else response[:200] + "..."


            if classification not in ["regional_bias", "non_regional_bias"]:
                # Fallback: Check entire response for keywords if structured parsing fails
                logger.warning(f"Invalid or missing classification from structured parse: '{classification}'. Raw response: '{response[:100]}...'")
                response_lower = response.lower()
                if "non_regional_bias" in response_lower or "non regional bias" in response_lower:
                    classification = "non_regional_bias"
                elif "regional_bias" in response_lower or "regional bias" in response_lower : # "regional_bias" alone
                    classification = "regional_bias"
                else: # Default if still not clear
                    logger.warning(f"Could not determine classification. Defaulting to non_regional_bias for safety.")
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

def save_results(results, output_dir, file_timestamp_or_name_part, prefix="model"):
    """Save classification results safely. Prefix is now more generic."""
    try:
        os.makedirs(output_dir, exist_ok=True)
        
        predictions_data = []
        for result in results:
            predictions_data.append({
                'index': result['index'],
                'comment': result['original_comment'],
                'prediction': result['classification'],
                'prediction_binary': 1 if result['classification'] == 'regional_bias' else 0 if result['classification'] == 'non_regional_bias' else -1,
                'reasoning': result.get('reasoning', '')
            })
        
        df_predictions = pd.DataFrame(predictions_data)
        # Use prefix in filename
        predictions_csv = os.path.join(output_dir, f"{prefix}_predictions_{file_timestamp_or_name_part}.csv")
        df_predictions.to_csv(predictions_csv, index=False)
        logger.info(f"Predictions saved to CSV: {predictions_csv}")
        
        try:
            json_file = os.path.join(output_dir, f"{prefix}_results_{file_timestamp_or_name_part}.json")
            with open(json_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            logger.info(f"Results saved to JSON: {json_file}")
        except Exception as e:
            logger.error(f"Error saving to JSON: {e}")
        
        return predictions_csv, df_predictions
        
    except Exception as e:
        logger.error(f"Error saving results: {e}")
        return None, None

def process_one_comment(classifier, comment, idx, max_comment_length=500):
    """Process a single comment with maximum error prevention"""
    try:
        free_memory()
        result = classifier.classify_with_cot(comment, max_length=max_comment_length)
        result['index'] = idx
        result['original_comment'] = comment
        pred_value = 1 if result['classification'] == 'regional_bias' else (0 if result['classification'] == 'non_regional_bias' else None)
        free_memory()
        return result, pred_value
    except Exception as e:
        logger.error(f"Error processing comment {idx}: {e}")
        return {
            "index": idx, "original_comment": comment, "classification": "error",
            "reasoning": f"Critical Error: {str(e)}", "full_response": ""
        }, None

def process_dataset(classifier, df, output_dir, save_interval=1, max_comment_length=500, run_timestamp="", model_slug="model"):
    """Process dataset one comment at a time with frequent saving"""
    results = []
    predictions = [] # Store 0 or 1 for metrics
    ground_truth = [] # Store 0 or 1 for metrics
    
    intermediary_dir = os.path.join(output_dir, "intermediary")
    os.makedirs(intermediary_dir, exist_ok=True)
    
    for idx_val, row_tuple in tqdm(df.iterrows(), total=len(df), desc="Processing comments"):
        # df.iterrows() returns (index, Series). We use the original DataFrame index.
        original_df_index = row_tuple.name # This is the original index from df before any slicing for chunks
        comment = str(row_tuple['Comment'])
        
        result, pred_value = process_one_comment(
            classifier, comment, original_df_index, max_comment_length=max_comment_length
        )
        results.append(result)

        current_gt_value = None
        if 'Level-1' in df.columns:
            try: current_gt_value = 1 if float(row_tuple['Level-1']) > 0 else 0
            except (ValueError, TypeError): logger.warning(f"Invalid Level-1 value at index {original_df_index}")
        elif 'Score' in df.columns: # Fallback to 'Score' if 'Level-1' not present
            try: current_gt_value = 1 if float(row_tuple['Score']) > 0 else 0
            except (ValueError, TypeError): logger.warning(f"Invalid Score value at index {original_df_index}")

        if pred_value is not None and current_gt_value is not None:
            predictions.append(pred_value)
            ground_truth.append(current_gt_value)
        elif pred_value is None and current_gt_value is not None:
            # If prediction failed but GT exists, we can't pair them for metrics. Log this.
             logger.warning(f"Prediction error for comment {original_df_index}, cannot include in metrics.")


        # Save intermediary results based on number of items processed in this call
        # (idx is 0-based count of items processed in current chunk/df)
        items_processed_in_call = len(results)
        if items_processed_in_call % save_interval == 0 or items_processed_in_call == len(df):
            intermediary_file = os.path.join(
                intermediary_dir, 
                f"{model_slug}_intermediary_progress_{items_processed_in_call}_of_{len(df)}_{run_timestamp}.csv"
            )
            df_intermediary = pd.DataFrame([
                {
                    'index': r['index'], 'comment': r['original_comment'], 'prediction': r['classification'],
                    'prediction_binary': 1 if r['classification'] == 'regional_bias' else 0 if r['classification'] == 'non_regional_bias' else -1,
                    'reasoning': r.get('reasoning', '')
                } for r in results # Save all results processed so far in this call
            ])
            df_intermediary.to_csv(intermediary_file, index=False)
            logger.info(f"Saved intermediary progress after {items_processed_in_call} comments to {intermediary_file}")
        
        free_memory()
    
    return results, predictions, ground_truth

def find_latest_checkpoint(output_dir, model_slug):
    """Find the latest checkpoint to resume from, specific to a model_slug."""
    # Look for chunk files for the specific model
    checkpoint_files = glob.glob(os.path.join(output_dir, "chunks", f"{model_slug}_predictions_*_chunk_*.csv"))
    if not checkpoint_files:
        return None, 0
    
    chunk_data = []
    for f_path in checkpoint_files:
        try:
            # Filename format: {model_slug}_predictions_{timestamp}_chunk_{chunk_num}.csv
            name_parts = Path(f_path).name.split('_')
            chunk_num_str = name_parts[-1].replace(".csv", "") # Should be chunk_num
            chunk_num = int(chunk_num_str)
            chunk_data.append({'chunk_num': chunk_num, 'file': f_path})
        except (IndexError, ValueError) as e:
            logger.warning(f"Could not parse chunk number from file {f_path}: {e}")
            continue
    
    if not chunk_data:
        return None, 0
    
    latest_chunk_info = max(chunk_data, key=lambda x: x['chunk_num'])
    latest_file = latest_chunk_info['file']
    latest_chunk_num = latest_chunk_info['chunk_num']
    logger.info(f"Found latest checkpoint file: {latest_file} (chunk {latest_chunk_num}) for model {model_slug}")
    
    # Consolidate all processed indices from ALL chunk files for this model_slug
    all_processed_indices = set()
    all_chunk_files = glob.glob(os.path.join(output_dir, "chunks", f"{model_slug}_predictions_*_chunk_*.csv"))
    for cf in all_chunk_files:
        try:
            df_checkpoint = pd.read_csv(cf)
            if 'index' in df_checkpoint.columns:
                all_processed_indices.update(df_checkpoint['index'].astype(int).tolist())
            else:
                logger.warning(f"Checkpoint file {cf} is missing 'index' column.")
        except Exception as e:
            logger.error(f"Error reading checkpoint file {cf}: {e}")

    return all_processed_indices, latest_chunk_num


def generate_evaluation_report(predictions, ground_truth, output_dir, timestamp, model_name):
    """Generate and save evaluation metrics and visualizations"""
    if not ground_truth or len(ground_truth) != len(predictions):
        logger.warning(f"Cannot generate evaluation report: ground truth (len {len(ground_truth)}) and predictions (len {len(predictions)}) mismatch or empty.")
        return
    
    try:
        model_short_name = get_model_slug(model_name) # Use slug for filenames
        
        report = classification_report(ground_truth, predictions, 
                                      target_names=['Non-Regional Bias', 'Regional Bias'],
                                      output_dict=True, zero_division=0)
        report_text = classification_report(ground_truth, predictions, 
                                          target_names=['Non-Regional Bias', 'Regional Bias'], zero_division=0)
        conf_matrix = confusion_matrix(ground_truth, predictions)
        
        logger.info("\n=== Classification Performance ===")
        logger.info(f"Model: {model_name}")
        logger.info("Classification Report:")
        logger.info(report_text)
        logger.info("Confusion Matrix:")
        logger.info(str(conf_matrix)) # Convert to string for logger
        
        report_file = os.path.join(output_dir, f"classification_report_{model_short_name}_{timestamp}.txt")
        with open(report_file, 'w') as f:
            f.write(f"=== Classification Report - {model_name} ===\n\n")
            f.write(f"Timestamp: {timestamp}\n")
            f.write(f"Total Samples Analyzed (with GT): {len(predictions)}\n\n")
            f.write(report_text)
            f.write("\n\n=== Confusion Matrix ===\n")
            f.write(str(conf_matrix))
            f.write("\nRows: Actual labels | Columns: Predicted labels\n")
        
        viz_dir = os.path.join(output_dir, "visualizations")
        os.makedirs(viz_dir, exist_ok=True)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['Non-Regional', 'Regional'],
                    yticklabels=['Non-Regional', 'Regional'])
        plt.title(f'Confusion Matrix - {model_short_name}')
        plt.ylabel('True Label'); plt.xlabel('Predicted Label')
        plt.tight_layout()
        cm_file = os.path.join(viz_dir, f"confusion_matrix_{model_short_name}_{timestamp}.png")
        plt.savefig(cm_file, dpi=300, bbox_inches='tight'); plt.close()
        
        report_data = {
            'timestamp': timestamp, 'model': model_name, 'total_samples_evaluated': len(predictions),
            'accuracy': report['accuracy'], 'metrics': report, 'confusion_matrix': conf_matrix.tolist()
        }
        json_report_file = os.path.join(output_dir, f"evaluation_report_{model_short_name}_{timestamp}.json")
        with open(json_report_file, 'w') as f: json.dump(report_data, f, indent=2)
        
        logger.info(f"Evaluation report saved: {report_file}")
        logger.info(f"Confusion matrix visualization: {cm_file}")
        
    except Exception as e:
        logger.error(f"Error generating evaluation report: {e}")


def main():
    args = parse_args()
    setup_environment(args) # Sets CUDA_VISIBLE_DEVICES, creates dirs, sets up logging
    
    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_slug = get_model_slug(args.model_name)

    logger.info(f"Starting regional bias classification with model: {args.model_name}")
    logger.info(f"Run timestamp: {run_timestamp}, Model slug for filenames: {model_slug}")
    logger.info(f"Arguments: {args}")

    try:
        data_path = Path(args.data_path)
        if not data_path.exists():
            logger.error(f"Input file not found: {data_path}"); return
        
        df_full = pd.read_csv(data_path)
        logger.info(f"Full dataset shape: {df_full.shape}, Columns: {df_full.columns.tolist()}")

        if 'Comment' not in df_full.columns:
            text_cols = [col for col in df_full.columns if df_full[col].dtype == 'object']
            if text_cols:
                logger.warning(f"'Comment' column not found. Using '{text_cols[0]}' as comment column.")
                df_full.rename(columns={text_cols[0]: 'Comment'}, inplace=True)
            else:
                raise ValueError("Dataset must contain a text column named 'Comment' or at least one object/string type column.")

        # Ensure unique index for processing and resuming
        df_full = df_full.reset_index(drop=True) 

        if args.max_samples and args.max_samples < len(df_full):
            logger.info(f"Limiting to first {args.max_samples} samples for processing.")
            df_to_process = df_full.iloc[:args.max_samples].copy()
        else:
            df_to_process = df_full.copy()
        
        processed_indices = set()
        last_chunk_num = 0
        if args.resume:
            logger.info("Attempting to resume from checkpoint...")
            # Pass model_slug to find_latest_checkpoint
            indices_from_ckpt, last_chunk_num_from_ckpt = find_latest_checkpoint(args.output_dir, model_slug)
            if indices_from_ckpt:
                processed_indices.update(indices_from_ckpt)
                last_chunk_num = last_chunk_num_from_ckpt
                logger.info(f"Resuming. Found {len(processed_indices)} already processed comments. Starting after chunk {last_chunk_num}.")
                # Filter df_to_process: keep rows whose original index is NOT in processed_indices
                df_to_process = df_to_process[~df_to_process.index.isin(processed_indices)].copy()
                logger.info(f"Remaining comments to process: {len(df_to_process)}")
            else:
                logger.info("No checkpoint found or checkpoint empty. Starting fresh.")
        
        if df_to_process.empty:
            logger.info("No new comments to process. Exiting.")
            # Optionally, consolidate all chunk results and generate final report if needed
            return

        try:
            classifier = RegionalBiasClassifier(
                model_name=args.model_name, cache_dir=args.cache_dir, offload_dir=args.offload_dir,
                quantization=args.quantization, cpu_offload=args.cpu_offload
            )
        except Exception as e:
            logger.error(f"Fatal error initializing classifier: {e}")
            return

        all_results_cumulative = []
        all_predictions_cumulative = []
        all_ground_truth_cumulative = []
        
        current_chunk_size = args.chunk_size
        
        for i in range(0, len(df_to_process), current_chunk_size):
            chunk_df = df_to_process.iloc[i : i + current_chunk_size].copy()
            current_process_chunk_num = last_chunk_num + (i // args.chunk_size) + 1 # Use original chunk_size for num
            logger.info(f"Processing chunk {current_process_chunk_num} (rows {chunk_df.index.min()} to {chunk_df.index.max()} of remaining data) with {len(chunk_df)} items.")
            
            try:
                chunk_results, chunk_predictions, chunk_gt = process_dataset(
                    classifier, chunk_df, args.output_dir,
                    save_interval=args.save_interval,
                    max_comment_length=args.max_comment_length,
                    run_timestamp=run_timestamp, # For intermediary files
                    model_slug=model_slug
                )
                all_results_cumulative.extend(chunk_results)
                all_predictions_cumulative.extend(chunk_predictions)
                all_ground_truth_cumulative.extend(chunk_gt)
                
                # Save results for this specific chunk
                chunk_output_dir = os.path.join(args.output_dir, "chunks")
                save_results(chunk_results, chunk_output_dir, 
                             f"{run_timestamp}_chunk_{current_process_chunk_num}", 
                             prefix=model_slug)
                logger.info(f"Successfully processed and saved chunk {current_process_chunk_num}.")
                free_memory()

            except torch.cuda.OutOfMemoryError as e_oom:
                logger.error(f"CUDA OOM in chunk {current_process_chunk_num}: {e_oom}. Model: {args.model_name}")
                logger.warning("Attempting to save progress made so far in this chunk if any, then exiting or reducing chunk size for NEXT run.")
                # Save any partial results from this failed chunk if `process_dataset` returned anything
                if chunk_results: # Assuming chunk_results might have partial data before OOM
                     save_results(chunk_results, chunk_output_dir, 
                                  f"{run_timestamp}_chunk_{current_process_chunk_num}_partial_OOM", 
                                  prefix=model_slug)
                # For simplicity, this version will stop. A more robust version might retry with smaller chunks.
                logger.error("Stopping due to OOM. Consider reducing chunk_size, max_comment_length, or using stronger quantization/offloading for the next run.")
                break # Exit the loop over chunks
            except Exception as e:
                logger.error(f"Error processing chunk {current_process_chunk_num}: {e}")
                logger.info("Skipping to next chunk if possible, or exiting if critical.")
                # Optionally save partials, then continue or break
                continue
        
        logger.info("Finished processing all designated chunks.")
        # Save final consolidated results (from all successfully processed chunks in this run)
        if all_results_cumulative:
            final_csv_path, _ = save_results(all_results_cumulative, args.output_dir, 
                                             f"{run_timestamp}_FINAL", prefix=model_slug)
            if final_csv_path:
                logger.info(f"Final consolidated results for this run saved to: {final_csv_path}")
            
            if all_predictions_cumulative and all_ground_truth_cumulative:
                 generate_evaluation_report(all_predictions_cumulative, all_ground_truth_cumulative, 
                                           args.output_dir, run_timestamp, args.model_name)
            else:
                logger.info("No ground truth or predictions available for final evaluation report for this run.")
        else:
            logger.info("No results were generated in this run.")

    except Exception as e:
        logger.error(f"Critical error in main execution: {e}", exc_info=True)
    finally:
        logger.info("Script execution finished.")
        free_memory() # Final cleanup

if __name__ == "__main__":
    # Initial GPU check before arg parsing (won't know specific GPU ID yet)
    if torch.cuda.is_available():
        logger.info(f"CUDA is available. Found {torch.cuda.device_count()} GPU(s).")
    else:
        logger.warning("CUDA is not available. Processing will be on CPU (expected to be very slow).")
    
    main()
