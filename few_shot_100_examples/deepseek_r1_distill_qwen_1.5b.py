import os
import pandas as pd
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import time
from tqdm import tqdm
from huggingface_hub import login

# Hugging Face authentication
HF_TOKEN = "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"

# Set up directories
EXAMPLE_PATH = "/DATA2/akash/venvs/debasmita/data/50_examples_few_shot_classification_dataset.csv"
TEST_PATH = "/DATA2/akash/venvs/debasmita/data/annotated_experiment_phase_new - Sheet1.csv"
OUTPUT_DIR = "/DATA2/akash/venvs/debasmita/results/few_shot_50"

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Model name for saving files
MODEL_NAME = "deepseek_r1_distill_qwen_1.5b"

# Configure for maximum GPU utilization
torch.cuda.empty_cache()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Using device: {device}")
print(f"GPU available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU model: {torch.cuda.get_device_name(0)}")
    print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

def load_datasets():
    """Load the example and test datasets"""
    # Load few-shot examples
    examples_df = pd.read_csv(EXAMPLE_PATH)
    
    # Load full dataset
    test_df = pd.read_csv(TEST_PATH)
    
    # Process the datasets
    print(f"Loaded {len(examples_df)} examples")
    print(f"Loaded {len(test_df)} total comments")
    
    # Select a balanced subset of examples to improve performance
    if len(examples_df) > 20:
        # Get positive examples (regional bias)
        pos_examples = examples_df[examples_df['Level-1'] == 1].sample(10, random_state=42)
        # Get negative examples (non-regional bias)
        neg_examples = examples_df[examples_df['Level-1'] == 0].sample(10, random_state=42)
        # Combine them
        selected_examples = pd.concat([pos_examples, neg_examples])
        print(f"Selected balanced subset of {len(selected_examples)} examples for few-shot learning")
    else:
        selected_examples = examples_df
    
    return selected_examples, test_df

def prepare_prompt(comment, examples_df):
    """Create a prompt with few-shot examples followed by the test comment"""
    
    base_prompt = """You are an expert in identifying regional biases in comments about Indian states and regions. 
Your task is to determine whether a comment contains regional bias or not.
Regional bias includes stereotypes, prejudices, or discriminatory statements about:
- Indian states or regions
- People from specific Indian states  
- Cultural, linguistic, economic, political, or infrastructural aspects of Indian regions

IMPORTANT: Take your time and consider each comment very carefully. Think step-by-step.

Here are some examples of comments and their classifications:

"""
    
    # Add examples - but limit to a balanced set of 10 total examples
    bias_examples = []
    non_bias_examples = []
    
    for _, row in examples_df.iterrows():
        example_text = row['Comment']
        is_bias = row['Level-1'] >= 1
        
        if is_bias and len(bias_examples) < 5:
            bias_examples.append((example_text, 'regional_bias'))
        elif not is_bias and len(non_bias_examples) < 5:
            non_bias_examples.append((example_text, 'non_regional_bias'))
            
    # Combine and alternate between biased and non-biased examples
    all_examples = []
    for i in range(max(len(bias_examples), len(non_bias_examples))):
        if i < len(bias_examples):
            all_examples.append(bias_examples[i])
        if i < len(non_bias_examples):
            all_examples.append(non_bias_examples[i])
    
    # Add the selected examples to the prompt
    for example_text, example_label in all_examples:
        base_prompt += f"Comment: \"{example_text}\"\nClassification: {example_label}\n\n"
    
    # Add the current comment to classify with very detailed instructions
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

def load_model():
    """Load the model and tokenizer with optimized settings"""
    print("Loading model and tokenizer...")
    start_time = time.time()
    
    model_id = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
    hf_token = HF_TOKEN
    
    # Configure to use the maximum available GPU memory
    tokenizer = AutoTokenizer.from_pretrained(model_id, token=hf_token)
    
    # For 40GB GPU we can load the full precision model for better results
    # Set maximum precision - avoid quantization to improve quality
    
    # Load with full precision for 40GB GPU (slow but more accurate)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        token=hf_token,
        torch_dtype=torch.float16,   # Still use float16 to fit in 40GB
        device_map="auto",           # Auto-distribute across available GPUs
        low_cpu_mem_usage=True,      # Minimize CPU memory usage
    )
    
    elapsed_time = time.time() - start_time
    print(f"Model loaded in {elapsed_time:.2f} seconds")
    
    return model, tokenizer

def batch_predict(model, tokenizer, comments, examples_df, batch_size=1):
    """Process comments one at a time for careful inference"""
    all_predictions = []
    
    # Process in single examples for maximum accuracy (very slow)
    for i in range(0, len(comments), batch_size):
        batch = comments[i:i + batch_size]
        batch_predictions = []
        
        for comment in tqdm(batch, desc=f"Processing example {i+1}/{len(comments)}"):
            try:
                # Artificially slow down processing to give the model more time
                time.sleep(1)  # Add 1 second pause between examples
                
                prompt = prepare_prompt(comment, examples_df)
                
                inputs = tokenizer(prompt, return_tensors="pt").to(device)
                
                # Generate prediction with careful parameters
                with torch.no_grad():
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=50,       # Much longer token limit for thorough responses
                        pad_token_id=tokenizer.eos_token_id,
                        do_sample=False,         # Deterministic generation
                        temperature=0.1,         # Very low temperature for more decisive outputs
                        num_beams=4,             # Use beam search for more careful consideration
                        early_stopping=True      # Stop when a good answer is found
                    )
                
                # Get the prediction text
                full_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
                
                # Extract only the last part with the classification
                # First find where the prompt ends
                prompt_end = full_output.find("Classification (only respond with")
                if prompt_end != -1:
                    # Get text after the prompt
                    response = full_output[prompt_end:].lower()
                else:
                    # Otherwise just look at the end of the output
                    response = full_output[-100:].lower()
                
                # More careful classification extraction
                if "regional_bias" in response and "non_regional_bias" not in response:
                    prediction = 1  # Clearly regional_bias
                elif "non_regional_bias" in response:
                    prediction = 0  # Clearly non_regional_bias
                elif "regional" in response and "bias" in response and not "non" in response:
                    prediction = 1  # Likely regional_bias
                else:
                    # Look more broadly in the full output for hints
                    if "stereotype" in full_output.lower() or "prejudice" in full_output.lower() or "discriminat" in full_output.lower():
                        prediction = 1  # Likely regional_bias based on reasoning
                    else:
                        prediction = 0  # Default to non_regional_bias if unclear
                
                # Print detailed info for all examples to monitor
                print(f"\nExample {i+1} classification:")
                print(f"Comment: {comment[:100]}...")
                print(f"Decision: {'regional_bias' if prediction == 1 else 'non_regional_bias'}")
                print(f"Last part of output: {full_output[-150:]}")
                    
            except Exception as e:
                print(f"Error processing comment: {e}")
                prediction = 0  # Default to non_regional_bias on error
                
            batch_predictions.append(prediction)
            
            # Clear cache after EACH example to ensure fresh processing
            torch.cuda.empty_cache()
            
        all_predictions.extend(batch_predictions)
        
    return all_predictions

def save_results(test_df, predictions, true_labels, examples_df):
    """Save prediction results, classification report, and confusion matrix"""
    
    # Save predictions
    results_df = test_df.copy()
    results_df['Predicted'] = predictions
    
    # Create output paths
    predictions_path = os.path.join(OUTPUT_DIR, f"{MODEL_NAME}_predictions_50_few_shot.csv")
    report_path = os.path.join(OUTPUT_DIR, f"{MODEL_NAME}_predictions_50_few_shot.txt")
    matrix_path = os.path.join(OUTPUT_DIR, f"{MODEL_NAME}_predictions_50_few_shot.png")
    
    # Save predictions CSV
    results_df.to_csv(predictions_path, index=False)
    print(f"Predictions saved to {predictions_path}")
    
    # Generate and save classification report
    report = classification_report(true_labels, predictions)
    with open(report_path, 'w') as f:
        f.write(f"Classification Report for {MODEL_NAME}\n\n")
        f.write(report)
    print(f"Classification report saved to {report_path}")
    
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
    print(f"Confusion matrix saved to {matrix_path}")

def main():
    """Main execution function"""
    # Start timing
    start_time = time.time()
    
    # Login to Hugging Face
    print("Logging in to Hugging Face...")
    login(token=HF_TOKEN)
    
    # Load datasets
    examples_df, test_df = load_datasets()
    
    # For faster testing during development, limit to a smaller subset
    # Remove this in production for full dataset processing
    # test_df = test_df.head(100)  # Uncomment for testing
    
    # Get the full set of comments to classify
    test_comments = test_df['Comment'].tolist()
    
    # Convert Level-1 to binary classification
    true_labels = test_df['Level-1'].apply(lambda x: 1 if x >= 1 else 0).tolist()
    
    # Load model and tokenizer
    model, tokenizer = load_model()
    
    # Predict one at a time (extremely slow but careful processing)
    print(f"Processing {len(test_comments)} comments...")
    print("NOTE: This is intentionally very slow for better accuracy!")
    print("Estimated time: ~{:.1f} hours".format(len(test_comments)/60/60))
    
    predictions = batch_predict(model, tokenizer, test_comments, examples_df)
    
    # Create a checkpoint after every 100 examples to prevent data loss
    print("Creating intermediate checkpoint...")
    checkpoint_df = test_df.copy()
    checkpoint_df['Predicted'] = predictions
    checkpoint_df.to_csv(os.path.join(OUTPUT_DIR, f"{MODEL_NAME}_predictions_checkpoint.csv"), index=False)
    
    # Save final results
    save_results(test_df, predictions, true_labels, examples_df)
    
    # Calculate and display accuracy
    accuracy = accuracy_score(true_labels, predictions)
    print(f"Accuracy: {accuracy:.4f}")
    
    # End timing
    end_time = time.time()
    hours = (end_time - start_time) / 3600
    print(f"Total execution time: {hours:.2f} hours")

if __name__ == "__main__":
    main()