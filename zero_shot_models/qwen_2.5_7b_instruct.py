import os
import sys
import pandas as pd
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from tqdm import tqdm
import json
from datetime import datetime
import gc
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from huggingface_hub import login
from collections import Counter
import logging
import argparse
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Zero-shot binary classification for regional bias detection using Mistral models')
    
    # Data and output paths
    parser.add_argument('--data-path', type=str, required=True,
                        help='Path to the annotated dataset CSV file')
    parser.add_argument('--output-dir', type=str, default='results/mistral/',
                        help='Directory to save output files') # Specify your output directory here
    parser.add_argument('--cache-dir', type=str, default=None,
                        help='Directory for model cache') # Specify cache directory here if needed
    parser.add_argument('--log-dir', type=str, default=None,
                        help='Directory for log files (defaults to output_dir)')
    
    # Model configuration
    parser.add_argument('--model-name', type=str, 
                        default='mistralai/Mistral-7B-Instruct-v0.3',
                        help='Model name or path')
    parser.add_argument('--gpu-id', type=int, default=0,
                        help='GPU ID to use for inference')
    parser.add_argument('--hf-token', type=str, default=None,
                        help='HuggingFace token for accessing gated models')
    
    # Memory optimization
    parser.add_argument('--quantization', type=str, default='none', 
                        choices=['4bit', '8bit', 'none'],
                        help='Quantization type (4bit, 8bit, or none)')
    parser.add_argument('--max-gpu-memory', type=str, default=None,
                        help='Maximum GPU memory to use (e.g., "16GB")')
    parser.add_argument('--cpu-offload', action='store_true', 
                        help='Enable CPU offloading for parts of the model')
    
    # Execution parameters
    parser.add_argument('--num-iterations', type=int, default=3,
                        help='Number of classification iterations to run')
    parser.add_argument('--max-samples', type=int, default=None,
                        help='Maximum number of samples to process (for testing)')
    parser.add_argument('--max-length', type=int, default=2048,
                        help='Maximum context length for tokenizer')
    parser.add_argument('--max-new-tokens', type=int, default=300,
                        help='Maximum number of new tokens to generate')
    
    args = parser.parse_args()
    return args

def setup_environment(args):
    """Set up environment variables and directories"""
    # Set GPU device
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)
    
    # Set cache directory if provided
    if args.cache_dir:
        os.environ['HF_HOME'] = args.cache_dir
        os.environ['TRANSFORMERS_CACHE'] = args.cache_dir
        os.makedirs(args.cache_dir, exist_ok=True)
        logger.info(f"Using cache directory: {args.cache_dir}")
    
    # Create output and log directories
    os.makedirs(args.output_dir, exist_ok=True)
    
    log_dir = args.log_dir if args.log_dir else args.output_dir
    os.makedirs(log_dir, exist_ok=True)
    
    # Set up file logger
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_prefix = args.model_name.split('/')[-1].lower() if '/' in args.model_name else args.model_name.lower()
    log_file = os.path.join(log_dir, f"{model_prefix}_classification_{timestamp}.log")
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(file_handler)
    
    # Log system information
    logger.info(f"Python version: {sys.version}")
    logger.info(f"PyTorch version: {torch.__version__}")
    
    # Log GPU information
    if torch.cuda.is_available():
        logger.info(f"Using GPU {args.gpu_id}: {torch.cuda.get_device_name(0)}")
        logger.info(f"Total GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    else:
        logger.warning("No GPU available! This process will be extremely slow.")

class RegionalBiasClassifier:
    def __init__(self, model_name="mistralai/Mistral-7B-Instruct-v0.3", 
                 cache_dir=None, hf_token=None, quantization="none", 
                 max_gpu_memory=None, cpu_offload=False):
        """Initialize the classifier with the specified model"""
        logger.info(f"Initializing {model_name}")
        if torch.cuda.is_available():
            logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
            logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        else:
            logger.warning("CUDA not available. Using CPU (this will be very slow).")
        
        # Clear GPU memory before loading
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        gc.collect()
        
        # Configure tokenizer
        tokenizer_kwargs = {
            'padding_side': 'left'  # Better for decoding-only models
        }
        
        if cache_dir:
            tokenizer_kwargs['cache_dir'] = cache_dir
        if hf_token:
            tokenizer_kwargs['token'] = hf_token
        
        # Load tokenizer
        logger.info("Loading tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, **tokenizer_kwargs)
        
        # Set pad token if not available
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Configure model parameters
        model_kwargs = {
            'torch_dtype': torch.float16,
            'low_cpu_mem_usage': True,
        }
        
        if cache_dir:
            model_kwargs['cache_dir'] = cache_dir
        if hf_token:
            model_kwargs['token'] = hf_token
        
        # Configure device mapping and memory limits
        if torch.cuda.is_available():
            # Compute optimal memory limit
            available_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
            if max_gpu_memory:
                # If specified as string with unit (e.g., "16GB"), extract the number
                if isinstance(max_gpu_memory, str) and "GB" in max_gpu_memory:
                    max_memory = float(max_gpu_memory.replace("GB", ""))
                else:
                    max_memory = float(max_gpu_memory)
                
                # Ensure we don't exceed available memory
                max_memory = min(max_memory, available_memory * 0.9)  # Leave some buffer
            else:
                # Use 85% of available memory by default
                max_memory = available_memory * 0.85
            
            # Format for model loading
            max_memory_str = f"{int(max_memory)}GB"
            logger.info(f"Setting max GPU memory usage to {max_memory_str}")
            
            # Set device map to force everything on GPU if possible
            model_kwargs['device_map'] = {'': 0}  # Force to GPU
            model_kwargs['max_memory'] = {0: max_memory_str}
        
        # Add quantization if requested
        if quantization == "4bit":
            logger.info("Using 4-bit quantization")
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                llm_int8_enable_fp32_cpu_offload=cpu_offload
            )
            model_kwargs['quantization_config'] = bnb_config
        elif quantization == "8bit":
            logger.info("Using 8-bit quantization")
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
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        gc.collect()
        
        # Load model
        logger.info("Loading model...")
        self.model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
        
        # Ensure model is in evaluation mode
        self.model.eval()
        
        # Log memory usage
        if torch.cuda.is_available():
            logger.info(f"Model loaded. Memory allocated: {torch.cuda.memory_allocated(0) / 1e9:.2f} GB")
    
    def classify_with_cot(self, text, max_length=2048, max_new_tokens=300):
        """Classify using Chain-of-Thought prompting for better reasoning"""
        cot_prompt = f"""[INST] You are an expert in identifying regional biases in comments about Indian states and regions. 
Your task is to determine whether a comment contains regional bias or not.

Regional bias includes stereotypes, prejudices, or discriminatory statements about:
- Indian states or regions
- People from specific Indian states  
- Cultural, linguistic, economic, political, or infrastructural aspects of Indian regions

Please analyze the following comment step by step:

Comment: "{text}"

Step 1: First, identify if this comment mentions any Indian state, region, or people from specific regions.

Step 2: Check if the comment contains any of these elements:
- Stereotypical statements about people from a region
- Generalizations about a state or its people
- Discriminatory language targeting regional identity
- Prejudiced views about regional culture, language, or traditions
- Biased statements about economic or developmental status
- Political stereotypes associated with regions

Step 3: Determine if these elements, if present, constitute bias or are merely factual/neutral observations.

Step 4: Based on your analysis, classify this comment as:
- "regional_bias": If it contains prejudiced, stereotypical, or discriminatory content about Indian regions/states
- "non_regional_bias": If it's neutral, factual, or does not contain regional bias

Please provide your reasoning followed by your final classification.

Format your response as:
Reasoning: [Your step-by-step analysis]
Classification: [regional_bias/non_regional_bias] [/INST]
"""
        
        try:
            # Clear memory before tokenization
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            gc.collect()
            
            # Tokenize the prompt
            inputs = self.tokenizer(
                cot_prompt, 
                return_tensors="pt", 
                truncation=True, 
                max_length=max_length
            )
            
            # Move inputs to GPU if available
            if torch.cuda.is_available():
                inputs = {k: v.to("cuda") for k, v in inputs.items()}
            
            # Generate response
            with torch.no_grad():
                try:
                    outputs = self.model.generate(
                        **inputs,
                        max_new_tokens=max_new_tokens,
                        temperature=0.1,  # Low temperature for more deterministic outputs
                        do_sample=False,  # Deterministic generation
                        pad_token_id=self.tokenizer.pad_token_id,
                        use_cache=True
                    )
                    
                except torch.cuda.OutOfMemoryError:
                    logger.error("CUDA out of memory during generation")
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        torch.cuda.synchronize()
                    gc.collect()
                    
                    # Try with reduced parameters
                    logger.info("Retrying with reduced parameters...")
                    try:
                        outputs = self.model.generate(
                            **inputs,
                            max_new_tokens=max_new_tokens // 2,  # Cut maximum tokens in half
                            temperature=0.1,
                            do_sample=False,
                            pad_token_id=self.tokenizer.pad_token_id,
                            use_cache=True
                        )
                    except Exception as e:
                        logger.error(f"Failed on retry: {e}")
                        return {
                            "classification": "non_regional_bias",  # Default to non-bias on failure
                            "reasoning": "Error during model inference",
                            "full_response": ""
                        }
                except Exception as e:
                    logger.error(f"Error in model generation: {e}")
                    return {
                        "classification": "non_regional_bias",  # Default to non-bias on failure
                        "reasoning": f"Error during processing: {str(e)}",
                        "full_response": ""
                    }
            
            # Decode the response
            response = self.tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
            
            # Clear intermediate tensors
            del outputs, inputs
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            gc.collect()
            
            # Extract classification and reasoning
            return self._parse_response(response)
            
        except Exception as e:
            logger.error(f"Error in classification: {e}")
            return {
                "classification": "non_regional_bias",  # Default to non-bias on failure
                "reasoning": f"Error: {str(e)}",
                "full_response": ""
            }
    
    def _parse_response(self, response):
        """Parse the model's response to extract classification and reasoning"""
        lines = response.strip().split('\n')
        reasoning = ""
        classification = ""
        
        for line in lines:
            if line.startswith("Reasoning:"):
                reasoning = line.replace("Reasoning:", "").strip()
            elif line.startswith("Classification:"):
                classification = line.replace("Classification:", "").strip().lower()
        
        # Validate classification - ensure we have a valid category
        if "regional_bias" in classification and "non" not in classification:
            classification = "regional_bias"
        elif "non_regional_bias" in classification or "non-regional_bias" in classification:
            classification = "non_regional_bias"
        else:
            logger.warning(f"Invalid classification: {classification}")
            # Default to non_regional_bias rather than error for better downstream handling
            classification = "non_regional_bias"
        
        return {
            "classification": classification,
            "reasoning": reasoning,
            "full_response": response
        }

def save_iteration_results(iteration_results, output_dir, iteration_num, timestamp, prefix="mistral"):
    """Save results for a single iteration"""
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Save as JSON
    json_file = os.path.join(output_dir, f"{prefix}_iteration_{iteration_num}_results_{timestamp}.json")
    with open(json_file, 'w') as f:
        json.dump(iteration_results, f, indent=2)
    
    # Save as detailed CSV
    df_results = pd.DataFrame(iteration_results)
    csv_file = os.path.join(output_dir, f"{prefix}_iteration_{iteration_num}_detailed_{timestamp}.csv")
    df_results.to_csv(csv_file, index=False)
    
    # Save predictions CSV
    predictions_data = [{
        'index': result['index'],
        'comment': result['original_comment'],
        'prediction': result['classification'],
        'prediction_binary': 1 if result['classification'] == 'regional_bias' else 0 if result['classification'] == 'non_regional_bias' else -1,
        'reasoning': result.get('reasoning', '')
    } for result in iteration_results]
    
    df_predictions = pd.DataFrame(predictions_data)
    predictions_csv = os.path.join(output_dir, f"{prefix}_iteration_{iteration_num}_predictions_{timestamp}.csv")
    df_predictions.to_csv(predictions_csv, index=False)
    
    logger.info(f"Iteration {iteration_num} results saved to {predictions_csv}")
    
    return predictions_csv

def generate_evaluation_report(all_iterations_predictions, ground_truth, all_iterations_results, output_dir, model_name, prefix="mistral"):
    """Generate comprehensive evaluation report for multiple iterations"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Calculate final predictions using majority voting
    final_predictions = []
    for i in range(len(ground_truth)):
        predictions_for_sample = [predictions[i] for predictions in all_iterations_predictions if i < len(predictions)]
        if predictions_for_sample:
            final_predictions.append(Counter(predictions_for_sample).most_common(1)[0][0])
    
    # Calculate metrics
    report = classification_report(ground_truth, final_predictions, 
                                target_names=['Non-Regional Bias', 'Regional Bias'], 
                                output_dict=True)
    report_text = classification_report(ground_truth, final_predictions, 
                                      target_names=['Non-Regional Bias', 'Regional Bias'])
    conf_matrix = confusion_matrix(ground_truth, final_predictions)
    
    # Create short model name for file naming
    model_short_name = model_name.split('/')[-1] if '/' in model_name else model_name
    
    # Save classification report as text file
    report_file = os.path.join(output_dir, f"classification_report_{prefix}_{timestamp}.txt")
    with open(report_file, 'w') as f:
        f.write(f"=== Classification Report - {model_short_name} (Multiple Iterations with Majority Voting) ===\n\n")
        f.write(f"Model: {model_name}\n")
        f.write(f"Timestamp: {timestamp}\n")
        f.write(f"Total Samples: {len(final_predictions)}\n")
        f.write(f"Number of Iterations: {len(all_iterations_predictions)}\n\n")
        f.write(report_text)
        f.write("\n\n=== Confusion Matrix ===\n")
        f.write(str(conf_matrix))
        f.write("\n\nRows: Actual labels\n")
        f.write("Columns: Predicted labels\n")
        
        # Add iteration-wise metrics
        f.write("\n\n=== Iteration-wise Performance ===\n")
        for i, predictions in enumerate(all_iterations_predictions):
            iter_report = classification_report(ground_truth[:len(predictions)], predictions, 
                                              target_names=['Non-Regional Bias', 'Regional Bias'], 
                                              output_dict=True)
            f.write(f"\nIteration {i+1} Accuracy: {iter_report['accuracy']:.4f}")
    
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
    confusion_matrix_file = os.path.join(viz_dir, f"confusion_matrix_{prefix}_{timestamp}.png")
    plt.savefig(confusion_matrix_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Iteration Agreement Visualization
    plt.figure(figsize=(10, 6))
    agreement_data = []
    for i in range(len(ground_truth)):
        predictions_for_sample = [predictions[i] for predictions in all_iterations_predictions if i < len(predictions)]
        if predictions_for_sample:
            agreement = len(set(predictions_for_sample)) == 1
            agreement_data.append(1 if agreement else 0)
    
    agreement_rate = sum(agreement_data) / len(agreement_data) if agreement_data else 0
    plt.bar(['Agreement', 'Disagreement'], 
            [sum(agreement_data), len(agreement_data) - sum(agreement_data)], 
            color=['green', 'red'])
    plt.title(f'Iteration Agreement Rate: {agreement_rate:.2%}')
    plt.ylabel('Number of Samples')
    agreement_file = os.path.join(viz_dir, f"iteration_agreement_{prefix}_{timestamp}.png")
    plt.savefig(agreement_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Class Distribution
    plt.figure(figsize=(10, 6))
    class_counts = [
        sum(label == 0 for label in ground_truth),
        sum(label == 1 for label in ground_truth)
    ]
    pred_counts = [
        sum(pred == 0 for pred in final_predictions),
        sum(pred == 1 for pred in final_predictions)
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
    
    distribution_file = os.path.join(viz_dir, f"class_distribution_{prefix}_{timestamp}.png")
    plt.savefig(distribution_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Save detailed evaluation report as JSON
    report_data = {
        'timestamp': timestamp,
        'model': model_name,
        'total_samples': len(final_predictions),
        'accuracy': report['accuracy'],
        'metrics': {
            'non_regional_bias': report['Non-Regional Bias'],
            'regional_bias': report['Regional Bias'],
            'weighted_avg': report['weighted avg'],
            'macro_avg': report['macro avg']
        },
        'confusion_matrix': conf_matrix.tolist(),
        'predictions_distribution': {
            'non_regional_bias': int(np.sum(np.array(final_predictions) == 0)),
            'regional_bias': int(np.sum(np.array(final_predictions) == 1))
        },
        'iteration_agreement_rate': agreement_rate
    }
    
    json_report_file = os.path.join(output_dir, f"evaluation_report_{prefix}_{timestamp}.json")
    with open(json_report_file, 'w') as f:
        json.dump(report_data, f, indent=2)
    
    logger.info("\n=== Evaluation Report ===")
    logger.info(f"Model: {model_name}")
    logger.info(f"Total Samples: {len(final_predictions)}")
    logger.info(f"Accuracy: {report['accuracy']:.4f}")
    logger.info(f"Reports saved to {output_dir}")

def main():
    """Main execution function"""
    # Parse command line arguments
    args = parse_args()
    
    # Set up environment variables and logging
    setup_environment(args)
    
    # Create a model prefix for file naming
    model_prefix = args.model_name.split('/')[-1].lower() if '/' in args.model_name else args.model_name.lower()
    
    # Generate timestamp for this run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Verify input file exists
    data_path = Path(args.data_path)
    if not data_path.exists():
        logger.error(f"Input file not found: {data_path}")
        return
    
    # Load data
    logger.info(f"Loading data from {data_path}")
    df = pd.read_csv(data_path)
    logger.info(f"Data shape: {df.shape}")
    
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
        logger.info(f"Limiting to {args.max_samples} samples")
        df = df.iloc[:args.max_samples].copy()
        
    # Try to login to HuggingFace if token provided
    try:
        if args.hf_token:
            login(token=args.hf_token)
            logger.info("Successfully logged in to HuggingFace")
        elif os.environ.get("HF_TOKEN"):
            login(token=os.environ["HF_TOKEN"])
            logger.info("Successfully logged in to HuggingFace using environment token")
    except Exception as e:
        logger.warning(f"Could not log in to HuggingFace: {e}")
        logger.warning("Attempting to proceed without login...")
    
    # Initialize classifier
    try:
        classifier = RegionalBiasClassifier(
            model_name=args.model_name,
            cache_dir=args.cache_dir,
            hf_token=args.hf_token,
            quantization=args.quantization,
            max_gpu_memory=args.max_gpu_memory,
            cpu_offload=args.cpu_offload
        )
    except torch.cuda.OutOfMemoryError:
        logger.error("Not enough GPU memory for this model")
        logger.error("Try using 4-bit quantization, enabling CPU offloading, or a smaller model.")
        return
    except Exception as e:
        logger.error(f"Error initializing classifier: {e}")
        return
    
    # Store results for all iterations
    all_iterations_results = []
    all_iterations_predictions = []
    ground_truth = []
    iteration_csv_files = []
    
    logger.info(f"\nStarting classification with {args.num_iterations} iterations...")
    
    # Run multiple iterations
    for iteration in range(args.num_iterations):
        logger.info(f"\n=== Iteration {iteration + 1}/{args.num_iterations} ===")
        
        iteration_results = []
        iteration_predictions = []
        
        for idx, row in tqdm(df.iterrows(), total=len(df), desc=f"Iteration {iteration + 1}"):
            comment = str(row['Comment'])
            
            try:
                # Get classification with CoT
                result = classifier.classify_with_cot(
                    comment, 
                    max_length=args.max_length,
                    max_new_tokens=args.max_new_tokens
                )
                
                # Store results
                result['index'] = idx
                result['original_comment'] = comment
                result['iteration'] = iteration + 1
                iteration_results.append(result)
                
                # For evaluation
                if result['classification'] != 'error':
                    iteration_predictions.append(1 if result['classification'] == 'regional_bias' else 0)
                    
                    # Collect ground truth only in first iteration
                    if iteration == 0:
                        if 'Score' in df.columns:  # Assuming Score > 0 means regional bias
                            ground_truth.append(1 if float(row['Score']) > 0 else 0)
                        elif 'Level-1' in df.columns:  # Alternative ground truth column
                            ground_truth.append(1 if float(row['Level-1']) > 0 else 0)
                
            except Exception as e:
                logger.error(f"Error processing comment {idx} in iteration {iteration + 1}: {e}")
                iteration_results.append({
                    "index": idx,
                    "original_comment": comment,
                    "classification": "non_regional_bias",  # Default to non-bias on error
                    "reasoning": f"Error: {str(e)}",
                    "full_response": "",
                    "iteration": iteration + 1
                })
            
            # Clear GPU cache periodically
            if (idx + 1) % 10 == 0 and torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                gc.collect()
        
        # Store iteration results
        all_iterations_results.append(iteration_results)
        all_iterations_predictions.append(iteration_predictions)
        
        # Save iteration results
        iter_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        predictions_csv = save_iteration_results(
            iteration_results, 
            args.output_dir, 
            iteration + 1, 
            iter_timestamp,
            prefix=model_prefix
        )
        iteration_csv_files.append(predictions_csv)
        
        logger.info(f"Iteration {iteration + 1} completed.")
    
    # Combine results and calculate final predictions
    final_results = []
    for idx in range(len(df)):
        comment_results = []
        for iteration_results in all_iterations_results:
            for result in iteration_results:
                if result['index'] == idx:
                    comment_results.append(result)
        
        # Get classifications from all iterations
        classifications = [r['classification'] for r in comment_results if r['classification'] != 'error']
        
        # Use majority voting for final classification
        if classifications:
            final_classification = Counter(classifications).most_common(1)[0][0]
        else:
            final_classification = 'non_regional_bias'  # Default to non-bias if no valid classifications
        
        final_results.append({
            'index': idx,
            'original_comment': df.iloc[idx]['Comment'],
            'final_classification': final_classification,
            'iteration_classifications': classifications,
            'all_iteration_results': comment_results
        })
    
    # Save final combined results
    final_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Save final JSON
    final_results_file = os.path.join(args.output_dir, f"{model_prefix}_final_results_{final_timestamp}.json")
    with open(final_results_file, 'w') as f:
        json.dump(final_results, f, indent=2)
    
    # Save final combined CSV
    final_df_data = []
    for result in final_results:
        final_df_data.append({
            'index': result['index'],
            'comment': result['original_comment'],
            'final_classification': result['final_classification'],
            'final_classification_binary': 1 if result['final_classification'] == 'regional_bias' else 0 if result['final_classification'] == 'non_regional_bias' else -1,
            'iteration_1': result['iteration_classifications'][0] if len(result['iteration_classifications']) > 0 else 'N/A',
            'iteration_2': result['iteration_classifications'][1] if len(result['iteration_classifications']) > 1 else 'N/A',
            'iteration_3': result['iteration_classifications'][2] if len(result['iteration_classifications']) > 2 else 'N/A',
            'num_iterations_completed': len(result['iteration_classifications']),
            'all_agree': len(set(result['iteration_classifications'])) == 1 if result['iteration_classifications'] else False
        })
    
    final_df = pd.DataFrame(final_df_data)
    final_csv_file = os.path.join(args.output_dir, f"{model_prefix}_final_combined_{final_timestamp}.csv")
    final_df.to_csv(final_csv_file, index=False)
    
    logger.info(f"\nFinal combined results saved to:")
    logger.info(f"- JSON: {final_results_file}")
    logger.info(f"- CSV: {final_csv_file}")
    
    # Generate evaluation report if ground truth is available
    if ground_truth:
        generate_evaluation_report(
            all_iterations_predictions, 
            ground_truth, 
            all_iterations_results, 
            args.output_dir,
            args.model_name,
            prefix=model_prefix
        )
    else:
        logger.warning("No ground truth available for evaluation")

if __name__ == "__main__":
    # Check GPU availability
    if torch.cuda.is_available():
        print(f"CUDA is available. Found {torch.cuda.device_count()} GPU(s)")
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        
        # Set PyTorch settings
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    else:
        print("CUDA is not available. Using CPU only (will be very slow).")
    
    # Run classification
    print("Starting regional bias classification...")
    main()
