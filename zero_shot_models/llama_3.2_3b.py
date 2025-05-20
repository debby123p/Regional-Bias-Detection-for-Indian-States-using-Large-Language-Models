import os
import pandas as pd
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import json
import argparse
from datetime import datetime
import gc
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from huggingface_hub import login
from collections import Counter

"""
Zero-shot binary classification for regional bias detection using Llama-3.2-3B with 
multi-iteration majority voting approach.

This script implements a classifier that can identify regional bias in text
using the Llama-3.2-3B model, running multiple iterations and combining results
for improved accuracy.
"""

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Zero-shot binary classification for regional bias detection using Llama-3-2.3B')
    
    # Data and output paths
    parser.add_argument('--data_path', type=str, required=True,
                        help='Path to the annotated dataset CSV file')
    parser.add_argument('--output_dir', type=str, default='results',
                        help='Directory to save output files')
    parser.add_argument('--cache_dir', type=str, default=None,
                        help='Directory for model cache')
    
    # Model configuration
    parser.add_argument('--model_name', type=str, 
                        default='meta-llama/Llama-3.2-3B',
                        help='Model name or path (default: meta-llama/Llama-3.2-3B)')
    parser.add_argument('--gpu_id', type=int, default=0,
                        help='GPU ID to use for inference')
    parser.add_argument('--hf_token', type=str, default=None,
                        help='HuggingFace token for accessing Llama models (required)')
    
    # Execution parameters
    parser.add_argument('--num_iterations', type=int, default=3,
                        help='Number of classification iterations to run')
    parser.add_argument('--max_samples', type=int, default=None,
                        help='Maximum number of samples to process (for testing)')
    parser.add_argument('--max_length', type=int, default=2048,
                        help='Maximum context length for tokenizer')
    parser.add_argument('--max_new_tokens', type=int, default=256,
                        help='Maximum number of new tokens to generate')
    
    args = parser.parse_args()
    return args

class RegionalBiasClassifier:
    def __init__(self, model_name="meta-llama/Llama-3.2-3B", 
                 gpu_id=0, cache_dir=None, hf_token=None):
        """
        Initialize the classifier with Llama-3 model.
        
        Args:
            model_name: Name of the model to load
            gpu_id: GPU ID to use for computation
            cache_dir: Directory for model cache
            hf_token: HuggingFace token for accessing Llama models (required)
        """
        # Set GPU device
        self.gpu_id = gpu_id
        os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
        torch.cuda.set_device(0)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        print(f"Initializing {model_name}")
        if torch.cuda.is_available():
            print(f"Using GPU: {torch.cuda.get_device_name(0)}")
            print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        else:
            print("CUDA not available. Using CPU (this will be very slow).")
        
        # Clear GPU cache before loading
        torch.cuda.empty_cache()
        gc.collect()
        
        # Login to HuggingFace (required for Llama models)
        if hf_token:
            login(token=hf_token)
            print("Logged in to HuggingFace with provided token")
        elif os.environ.get("HF_TOKEN"):
            login(token=os.environ["HF_TOKEN"])
            print("Logged in to HuggingFace using environment token")
        else:
            print("WARNING: No HuggingFace token provided. Llama models require authentication.")
        
        # Initialize tokenizer
        tokenizer_kwargs = {
            'padding_side': 'left',
            'trust_remote_code': True
        }
        
        if cache_dir:
            tokenizer_kwargs['cache_dir'] = cache_dir
        if hf_token:
            tokenizer_kwargs['token'] = hf_token
            
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, **tokenizer_kwargs)
        
        # Ensure tokenizer has padding token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Set up model loading configuration - optimized for smaller Llama-3 model
        model_kwargs = {
            'torch_dtype': torch.float16,
            'low_cpu_mem_usage': True,
            'trust_remote_code': True
        }
        
        if cache_dir:
            model_kwargs['cache_dir'] = cache_dir
        if hf_token:
            model_kwargs['token'] = hf_token
            
        # Configure device mapping for GPU - Llama-3.2-3B is small enough for efficient memory usage
        if torch.cuda.is_available():
            available_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
            
            # Llama-3.2-3B is approximately 6.7GB in half precision, so we can be more aggressive with memory
            max_memory_str = f"{int(available_memory * 0.95)}GB"  # Use up to 95% of available memory
            
            model_kwargs['device_map'] = {'': 0}  # Force everything to specified GPU
            model_kwargs['max_memory'] = {0: max_memory_str}
            
            print(f"Setting max GPU memory usage to {max_memory_str}")
        
        # Load the model
        self.model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
        self.model.eval()
        
        if torch.cuda.is_available():
            print(f"Model loaded. Memory allocated: {torch.cuda.memory_allocated(0) / 1e9:.2f} GB")
        
    def classify_with_cot(self, text, max_length=2048, max_new_tokens=256):
        """
        Classify using Chain-of-Thought prompting with Llama-3 format
        
        Args:
            text: Text to classify
            max_length: Maximum context length for tokenization
            max_new_tokens: Maximum number of new tokens to generate
            
        Returns:
            Dictionary containing classification results
        """
        # Llama-3.2 specific prompt format
        cot_prompt = f"""<|begin_of_text|><|system|>
You are an expert in identifying regional biases in comments about Indian states and regions.
<|end_of_turn|>
<|user|>
I need your help to analyze if this comment contains regional bias:

"{text}"

Regional bias includes stereotypes, prejudices, or discriminatory statements about:
- Indian states or regions
- People from specific Indian states  
- Cultural, linguistic, economic, political, or infrastructural aspects of Indian regions

Please follow these steps in your analysis:

1. First, identify if this comment mentions any Indian state, region, or people from specific regions.

2. Check if the comment contains any of these elements:
- Stereotypical statements about people from a region
- Generalizations about a state or its people
- Discriminatory language targeting regional identity
- Prejudiced views about regional culture, language, or traditions
- Biased statements about economic or developmental status
- Political stereotypes associated with regions

3. Determine if these elements, if present, constitute bias or are merely factual/neutral observations.

4. Based on your analysis, classify this comment as:
- "regional_bias": If it contains prejudiced, stereotypical, or discriminatory content about Indian regions/states
- "non_regional_bias": If it's neutral, factual, or does not contain regional bias

Format your response as:
Reasoning: [Your step-by-step analysis]
Classification: [regional_bias/non_regional_bias]
<|end_of_turn|>
<|assistant|>
"""
        
        # Tokenize the prompt
        inputs = self.tokenizer(cot_prompt, return_tensors="pt", truncation=True, max_length=max_length)
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        
        # Generate response
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=0.1,  # Low temperature for deterministic outputs
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id,
                use_cache=True
            )
        
        # Decode the response
        response = self.tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
        
        # Clear memory
        del outputs, inputs
        torch.cuda.empty_cache()
        
        # Parse the response to extract reasoning and classification
        lines = response.strip().split('\n')
        reasoning = ""
        classification = ""
        
        for line in lines:
            if line.startswith("Reasoning:"):
                reasoning = line.replace("Reasoning:", "").strip()
            elif line.startswith("Classification:"):
                classification = line.replace("Classification:", "").strip().lower()
        
        # Validate classification
        if "regional_bias" in classification:
            classification = "regional_bias"
        elif "non_regional_bias" in classification or "non-regional_bias" in classification:
            classification = "non_regional_bias"
        else:
            print(f"Warning: Invalid classification found: {classification}")
            classification = "error"
        
        return {
            "classification": classification,
            "reasoning": reasoning,
            "full_response": response
        }

def save_iteration_results(iteration_results, output_dir, iteration_num, timestamp, model_prefix="llama"):
    """
    Save results for a single iteration
    
    Args:
        iteration_results: List of result dictionaries for this iteration
        output_dir: Directory to save results
        iteration_num: Current iteration number
        timestamp: Timestamp for file naming
        model_prefix: Model name prefix for file naming
    
    Returns:
        Path to the saved predictions CSV file
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract prediction data
    predictions_data = []
    for result in iteration_results:
        predictions_data.append({
            'index': result['index'],
            'comment': result['original_comment'],
            'prediction': result['classification'],
            'prediction_binary': 1 if result['classification'] == 'regional_bias' else 0,
            'reasoning': result.get('reasoning', '')
        })
    
    # Save predictions CSV
    df_predictions = pd.DataFrame(predictions_data)
    predictions_csv = os.path.join(output_dir, f"{model_prefix}_iteration_{iteration_num}_predictions_{timestamp}.csv")
    df_predictions.to_csv(predictions_csv, index=False)
    
    print(f"Iteration {iteration_num} results saved to: {predictions_csv}")
    
    return predictions_csv

def generate_evaluation_report(all_iterations_predictions, ground_truth, output_dir, model_name, model_prefix="llama"):
    """
    Generate evaluation report for multiple iterations with majority voting
    
    Args:
        all_iterations_predictions: List of prediction lists for each iteration
        ground_truth: List of ground truth labels
        output_dir: Directory to save evaluation results
        model_name: Full model name for report headers
        model_prefix: Model name prefix for file naming
    """
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
    
    # Print evaluation results
    print("\n=== Classification Performance ===")
    print(f"Accuracy: {report['accuracy']:.4f}")
    print("\nClassification Report:")
    print(report_text)
    print("\nConfusion Matrix:")
    print(conf_matrix)
    
    # Create visualizations directory
    viz_dir = os.path.join(output_dir, "visualizations")
    os.makedirs(viz_dir, exist_ok=True)
    
    # 1. Confusion Matrix visualization
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Non-Regional', 'Regional'],
                yticklabels=['Non-Regional', 'Regional'])
    plt.title('Confusion Matrix - Majority Voting from Multiple Iterations')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    confusion_matrix_file = os.path.join(viz_dir, f"confusion_matrix_{model_prefix}_{timestamp}.png")
    plt.savefig(confusion_matrix_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Class Distribution visualization
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
    
    distribution_file = os.path.join(viz_dir, f"class_distribution_{model_prefix}_{timestamp}.png")
    plt.savefig(distribution_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Evaluation visualizations saved to: {viz_dir}")

def main():
    """Main execution function"""
    # Parse command line arguments
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Extract model short name for naming files
    model_short_name = args.model_name.split('/')[-1] if '/' in args.model_name else args.model_name
    model_prefix = "llama3-2-3b"  # Specific for Llama-3.2-3B
    
    # Print execution parameters
    print("=== Regional Bias Classification with Llama-3.2-3B ===")
    print(f"Model: {args.model_name}")
    print(f"Data path: {args.data_path}")
    print(f"Number of iterations: {args.num_iterations}")
    
    # Check GPU availability
    if not torch.cuda.is_available():
        print("Warning: No GPU available! This will be very slow.")
    else:
        print(f"Using GPU {args.gpu_id}: {torch.cuda.get_device_name(args.gpu_id)}")
    
    # Load data
    print(f"Loading data from {args.data_path}")
    df = pd.read_csv(args.data_path)
    print(f"Data shape: {df.shape}")
    
    # Apply max_samples limit if specified
    if args.max_samples and args.max_samples < len(df):
        print(f"Limiting to {args.max_samples} samples for processing")
        df = df.iloc[:args.max_samples]
    
    # Check for HuggingFace token (required for Llama models)
    hf_token = args.hf_token or os.environ.get("HF_TOKEN")
    if not hf_token:
        print("ERROR: HuggingFace token is required for Llama-3.2-3B model.")
        print("Please provide a token via --hf_token or set the HF_TOKEN environment variable.")
        return
    
    # Initialize classifier
    classifier = RegionalBiasClassifier(
        model_name=args.model_name,
        gpu_id=args.gpu_id,
        cache_dir=args.cache_dir,
        hf_token=hf_token
    )
    
    # Store results for all iterations
    all_iterations_results = []
    all_iterations_predictions = []
    ground_truth = []
    iteration_csv_files = []
    
    print(f"\nStarting classification with {args.num_iterations} iterations...")
    
    # Run multiple iterations
    for iteration in range(args.num_iterations):
        print(f"\n=== Iteration {iteration + 1}/{args.num_iterations} ===")
        
        iteration_results = []
        iteration_predictions = []
        
        # Process each comment
        for idx, row in tqdm(df.iterrows(), total=len(df), desc=f"Iteration {iteration + 1}"):
            comment = str(row['Comment'])
            
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
            
            # Clear GPU cache periodically - Llama-3.2-3B has smaller memory footprint
            if (idx + 1) % 30 == 0:
                torch.cuda.empty_cache()
                gc.collect()
        
        # Store iteration results
        all_iterations_results.append(iteration_results)
        all_iterations_predictions.append(iteration_predictions)
        
        # Save iteration results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        predictions_csv = save_iteration_results(
            iteration_results, args.output_dir, 
            iteration + 1, timestamp, model_prefix
        )
        iteration_csv_files.append(predictions_csv)
        
        print(f"Iteration {iteration + 1} completed.")
        
        # Clear memory between iterations
        torch.cuda.empty_cache()
        gc.collect()
    
    # Calculate majority vote for final predictions
    final_results = []
    for idx in range(len(df)):
        # Collect all classifications for this comment
        classifications = []
        for iteration_results in all_iterations_results:
            for result in iteration_results:
                if result['index'] == idx and result['classification'] != 'error':
                    classifications.append(result['classification'])
        
        # Use majority voting for final classification
        if classifications:
            final_classification = Counter(classifications).most_common(1)[0][0]
        else:
            final_classification = 'error'
        
        final_results.append({
            'index': idx,
            'comment': df.iloc[idx]['Comment'],
            'final_classification': final_classification,
            'all_classifications': classifications,
            'agreement': len(set(classifications)) == 1 if classifications else False
        })
    
    # Save final combined results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    final_df = pd.DataFrame(final_results)
    final_csv_file = os.path.join(args.output_dir, f"{model_prefix}_final_combined_{timestamp}.csv")
    final_df.to_csv(final_csv_file, index=False)
    
    print(f"\nFinal combined results saved to: {final_csv_file}")
    
    # Generate evaluation report if ground truth is available
    if ground_truth:
        generate_evaluation_report(
            all_iterations_predictions, ground_truth, 
            args.output_dir, args.model_name, model_prefix
        )
    else:
        print("No ground truth available for evaluation")

if __name__ == "__main__":
    main()
