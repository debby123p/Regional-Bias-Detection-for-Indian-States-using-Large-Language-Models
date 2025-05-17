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
        description='Zero-shot binary classification for regional bias detection using DeepSeek models with memory optimization')
    
    # Data and output paths
    parser.add_argument('--data-path', type=str, required=True,
                        help='Path to the annotated dataset CSV file')
    parser.add_argument('--output-dir', type=str, default='results/deepseek/',
                        help='Directory to save output files')
    parser.add_argument('--cache-dir', type=str, default=None,
                        help='Directory for model cache')
    parser.add_argument('--offload-dir', type=str, default=None,
                        help='Directory for model offloading')
    
    # Model configuration
    parser.add_argument('--model-name', type=str, 
                        default='deepseek-ai/DeepSeek-R1-Distill-Qwen-7B',
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
                        help='Interval to save results during processing')
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
    log_file = os.path.join(args.output_dir, f"deepseek_classification_{timestamp}.log")
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
        logger.info(f"Using GPU {args.gpu_id}: {torch.cuda.get_device_name(0)}")
        logger.info(f"Total GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        
        # Set additional PyTorch settings for memory optimization
        torch.backends.cudnn.benchmark = False  # Disable for memory savings
        torch.backends.cudnn.deterministic = True  # Enable for memory savings
    else:
        logger.warning("No GPU available! Processing will be extremely slow.")

def free_memory():
    """Aggressively free GPU memory"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    gc.collect()

class RegionalBiasClassifier:
    def __init__(self, model_name="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B", 
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
        logger.info(f"Initializing {model_name} with extreme memory optimizations")
        if torch.cuda.is_available():
            logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
            logger.info(f"Total GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        
        # Aggressively free memory before loading
        free_memory()
        
        try:
            # Load tokenizer first
            logger.info("Loading tokenizer only first...")
            tokenizer_kwargs = {
                'trust_remote_code': True,
                'padding_side': "left",
                'local_files_only': False  # Allow downloading if needed
            }
            
            if cache_dir:
                tokenizer_kwargs['cache_dir'] = cache_dir
                
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, **tokenizer_kwargs)
            
            # Set pad token if not available
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
            
            # Configure quantization based on settings
            model_kwargs = {
                'trust_remote_code': True,
                'low_cpu_mem_usage': True,
                'torch_dtype': torch.float16,
            }
            
            if cache_dir:
                model_kwargs['cache_dir'] = cache_dir
                
            if offload_dir:
                model_kwargs['offload_folder'] = offload_dir
                model_kwargs['offload_state_dict'] = True
            
            # Configure quantization
            if quantization == "4bit":
                logger.info("Using 4-bit quantization for model loading")
                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,  # 4-bit quantization is essential for low memory
                    bnb_4bit_use_double_quant=True,  # Use double quantization to further compress
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.float16,
                    llm_int8_enable_fp32_cpu_offload=cpu_offload  # Enable CPU offloading if requested
                )
                model_kwargs['quantization_config'] = bnb_config
            elif quantization == "8bit":
                logger.info("Using 8-bit quantization for model loading")
                bnb_config = BitsAndBytesConfig(
                    load_in_8bit=True,
                    bnb_8bit_use_double_quant=True,
                    bnb_8bit_compute_dtype=torch.float16,
                    llm_int8_enable_fp32_cpu_offload=cpu_offload
                )
                model_kwargs['quantization_config'] = bnb_config
            else:
                logger.info("Using float16 precision (no quantization)")
            
            # Clear memory before model loading
            free_memory()
            
            # Load model with memory-optimized settings
            logger.info("Loading model with extreme memory optimizations...")
            
            # Determine device mapping strategy
            if torch.cuda.is_available():
                logger.info("Using 'auto' device mapping for optimal memory usage")
                model_kwargs['device_map'] = "auto"
            
            self.model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
            
            # Set to evaluation mode
            self.model.eval()
            
            # Free memory again
            free_memory()
            
            if torch.cuda.is_available():
                logger.info(f"Model loaded. Memory allocated: {torch.cuda.memory_allocated(0) / 1e9:.2f} GB")
                logger.info(f"Memory reserved: {torch.cuda.memory_reserved(0) / 1e9:.2f} GB")
            
        except Exception as e:
            logger.error(f"Error initializing model: {e}")
            raise
    
    def classify_with_cot(self, text, max_length=500):
        """
        Classify using Chain-of-Thought prompting with limited context size
        
        Args:
            text: Text to classify
            max_length: Maximum length of text to process
            
        Returns:
            Dictionary containing classification results
        """
        # Truncate text to save memory
        if len(text) > max_length:
            text = text[:max_length] + "..."
        
        # DeepSeek-specific prompt format
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
                max_length=768,  # Reduced context size for memory saving
                padding=True
            )
            
            # Move inputs to GPU if available
            if torch.cuda.is_available():
                inputs = {k: v.to("cuda") for k, v in inputs.items()}
            
            # Free memory again
            free_memory()
            
            with torch.no_grad():
                try:
                    # Ultra-conservative generation parameters
                    generation_params = {
                        "max_new_tokens": 100,  # Severely limited output
                        "temperature": 0.1,  # Low temperature for more deterministic output
                        "do_sample": False,  # Use greedy decoding to save memory
                        "pad_token_id": self.tokenizer.pad_token_id,
                        "eos_token_id": self.tokenizer.eos_token_id,
                        "repetition_penalty": 1.0
                    }
                    
                    # Try generating with conservative settings
                    outputs = self.model.generate(**inputs, **generation_params)
                        
                except torch.cuda.OutOfMemoryError as e:
                    logger.error(f"CUDA out of memory: {e}")
                    free_memory()
                    
                    # Emergency fallback mode - even more restricted
                    logger.info("Retrying with more restricted generation parameters")
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

def process_one_comment(classifier, comment, idx, max_comment_length=500):
    """Process a single comment with maximum error prevention"""
    try:
        # Aggressively free memory before processing
        free_memory()
        
        # Get classification
        result = classifier.classify_with_cot(comment, max_length=max_comment_length)
        
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

def process_dataset(classifier, df, output_dir, save_interval=1, max_comment_length=500):
    """Process dataset one comment at a time with frequent saving"""
    results = []
    predictions = []
    ground_truth = []
    
    # Create timestamp for saving
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Output directory for intermediary results
    intermediary_dir = os.path.join(output_dir, "intermediary")
    os.makedirs(intermediary_dir, exist_ok=True)
    
    # Process each comment individually to minimize memory usage
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing comments"):
        comment = str(row['Comment'])
        
        # Process comment
        result, pred_value = process_one_comment(
            classifier, comment, idx, max_comment_length=max_comment_length
        )
        
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
        
        # Save results after every specified interval
        if (idx + 1) % save_interval == 0 or idx == len(df) - 1:
            # Save intermediary results
            intermediary_file = os.path.join(
                intermediary_dir, 
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

def generate_evaluation_report(predictions, ground_truth, output_dir, timestamp, model_name):
    """Generate and save evaluation metrics and visualizations"""
    if not ground_truth or len(ground_truth) != len(predictions):
        logger.warning("Cannot generate evaluation report: ground truth missing or mismatched")
        return
    
    try:
        # Create short model name for file naming
        model_short_name = model_name.split('/')[-1] if '/' in model_name else model_name
        
        # Compute metrics
        report = classification_report(ground_truth, predictions, 
                                      target_names=['Non-Regional Bias', 'Regional Bias'],
                                      output_dict=True)
        report_text = classification_report(ground_truth, predictions, 
                                          target_names=['Non-Regional Bias', 'Regional Bias'])
        conf_matrix = confusion_matrix(ground_truth, predictions)
        
        # Log metrics
        logger.info("\n=== Classification Performance ===")
        logger.info("Classification Report:")
        logger.info(report_text)
        logger.info("Confusion Matrix:")
        logger.info(conf_matrix)
        
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
        
        # 1. Confusion Matrix
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
        
        # 2. Class Distribution
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
        
        # Save detailed evaluation report as JSON
        report_data = {
            'timestamp': timestamp,
            'model': model_name,
            'total_samples': len(predictions),
            'accuracy': report['accuracy'],
            'metrics': {
                'non_regional_bias': report['Non-Regional Bias'],
                'regional_bias': report['Regional Bias'],
                'weighted_avg': report['weighted avg'],
                'macro_avg': report['macro avg']
            },
            'confusion_matrix': conf_matrix.tolist(),
            'predictions_distribution': {
                'non_regional_bias': sum(pred == 0 for pred in predictions),
                'regional_bias': sum(pred == 1 for pred in predictions)
            }
        }
        
        json_report_file = os.path.join(output_dir, f"evaluation_report_{timestamp}.json")
        with open(json_report_file, 'w') as f:
            json.dump(report_data, f, indent=2)
        
        logger.info(f"Evaluation report saved: {report_file}")
        logger.info(f"Confusion matrix visualization: {confusion_matrix_file}")
        logger.info(f"Class distribution visualization: {distribution_file}")
        
    except Exception as e:
        logger.error(f"Error generating evaluation report: {e}")

def main():
    """Main execution function with a focus on memory optimization"""
    # Parse command line arguments
    args = parse_args()
    
    # Set up environment variables and logging
    setup_environment(args)
    
    # Generate timestamp for this run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    try:
        # Verify input file exists
        data_path = Path(args.data_path)
        if not data_path.exists():
            logger.error(f"Input file not found: {data_path}")
            return
        
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
        
        # Apply max_samples limit if specified
        if args.max_samples and args.max_samples < len(df):
            logger.info(f"Limiting to {args.max_samples} samples for processing")
            df = df.iloc[:args.max_samples].copy()
        
        # Check for resume
        processed_indices = None
        last_chunk_num = 0
        if args.resume:
            processed_indices, last_chunk_num = find_latest_checkpoint(args.output_dir)
            if processed_indices:
                logger.info(f"Resuming from chunk {last_chunk_num}, {len(processed_indices)} comments already processed")
                # Filter out already processed rows
                df = df[~df.index.isin(processed_indices)].copy()
                logger.info(f"Remaining comments to process: {len(df)}")
        
        # Initialize classifier with memory optimizations
        try:
            classifier = RegionalBiasClassifier(
                model_name=args.model_name,
                cache_dir=args.cache_dir,
                offload_dir=args.offload_dir,
                quantization=args.quantization,
                cpu_offload=args.cpu_offload
            )
        except torch.cuda.OutOfMemoryError as e:
            logger.error(f"CUDA out of memory during model initialization: {e}")
            logger.error("Not enough GPU memory for this model with current settings.")
            logger.error("Try using 4-bit quantization, enabling CPU offloading, or a smaller model.")
            return
        except Exception as e:
            logger.error(f"Error initializing classifier: {e}")
            return
        
        # Process in chunks to avoid memory issues
        total_rows = len(df)
        all_results = []
        all_predictions = []
        all_ground_truth = []
        chunk_size = args.chunk_size
        
        logger.info(f"Processing {total_rows} comments in chunks of {chunk_size}...")
        
        # Process in chunks
        for chunk_start in range(0, total_rows, chunk_size):
            chunk_end = min(chunk_start + chunk_size, total_rows)
            chunk_num = last_chunk_num + (chunk_start // chunk_size) + 1
            logger.info(f"Processing chunk {chunk_num}: rows {chunk_start} to {chunk_end-1}")
            
            # Process chunk
            chunk_df = df.iloc[chunk_start:chunk_end].copy()
            
            try:
                # Process dataset with frequent saving
                chunk_results, chunk_predictions, chunk_ground_truth = process_dataset(
                    classifier,
                    chunk_df,
                    args.output_dir,
                    save_interval=args.save_interval,
                    max_comment_length=args.max_comment_length
                )
                
                # Add to cumulative results
                all_results.extend(chunk_results)
                if chunk_predictions:
                    all_predictions.extend(chunk_predictions)
                if chunk_ground_truth:
                    all_ground_truth.extend(chunk_ground_truth)
                
                # Save intermediate results after each chunk
                chunk_output_dir = os.path.join(args.output_dir, "chunks")
                os.makedirs(chunk_output_dir, exist_ok=True)
                save_results(
                    chunk_results, 
                    chunk_output_dir, 
                    f"{timestamp}_chunk_{chunk_num}",
                    prefix="deepseek"
                )
                
                # Free memory after processing chunk
                free_memory()
                
            except torch.cuda.OutOfMemoryError as e:
                logger.error(f"CUDA out of memory during chunk processing: {e}")
                logger.warning(f"Reducing chunk size and retrying for remaining chunks...")
                
                # Save what we have so far before adjusting
                save_results(
                    all_results, 
                    args.output_dir, 
                    f"{timestamp}_partial_{chunk_start}", 
                    prefix="deepseek"
                )
                
                # Reduce chunk size for future chunks
                new_chunk_size = max(10, chunk_size // 2)
                logger.info(f"Reducing chunk size from {chunk_size} to {new_chunk_size}")
                chunk_size = new_chunk_size
                
                # Process remaining data with smaller chunk size
                continue
                
            except Exception as e:
                logger.error(f"Error processing chunk {chunk_num}: {e}")
                # Save what we have so far
                save_results(
                    all_results, 
                    args.output_dir, 
                    f"{timestamp}_partial_{chunk_start}", 
                    prefix="deepseek"
                )
                continue
            
            # Success message for chunk
            logger.info(f"Successfully processed chunk {chunk_num}")
        
        # Save final complete results
        final_csv, final_df = save_results(
            all_results, 
            args.output_dir, 
            timestamp, 
            prefix="deepseek"
        )
        
        # Generate summary statistics
        if all_predictions:
            regional_bias_count = all_predictions.count(1)
            non_regional_bias_count = all_predictions.count(0)
            
            logger.info("\n=== Classification Summary ===")
            logger.info(f"Total comments classified: {len(all_predictions)}")
            logger.info(f"Regional bias comments: {regional_bias_count} ({regional_bias_count/len(all_predictions)*100:.2f}%)")
            logger.info(f"Non-regional bias comments: {non_regional_bias_count} ({non_regional_bias_count/len(all_predictions)*100:.2f}%)")
        
        logger.info(f"Complete results saved to: {final_csv}")
        
        # Generate evaluation report if ground truth is available
        if all_ground_truth and len(all_ground_truth) == len(all_predictions):
            generate_evaluation_report(
                all_predictions, 
                all_ground_truth, 
                args.output_dir, 
                timestamp, 
                args.model_name
            )
        
    except Exception as e:
        logger.error(f"Error in main execution: {e}")
        raise

if __name__ == "__main__":
    # Check for GPU and set optimizations
    if torch.cuda.is_available():
        logger.info(f"CUDA is available. Found {torch.cuda.device_count()} GPU(s)")
        logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
        
        # Optimize PyTorch for memory efficiency
        torch.backends.cudnn.benchmark = False  # Disable for memory savings
        torch.backends.cudnn.deterministic = True  # Enable for memory savings
    else:
        logger.warning("CUDA is not available. Using CPU only (will be very slow).")
    
    # Run classifier
    logger.info("Starting regional bias classification with DeepSeek model...")
    main()
