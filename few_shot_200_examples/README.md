## Regional Bias Detection - 200 Few-Shot Examples

This contains code for detecting regional biases in text comments using state-of-the-art language models with a 200-example few-shot learning approach.

## Overview
The code implements a few-shot learning pipeline where 200 annotated examples (100 with regional bias, 100 without) are used to prompt large language models to classify new comments. This approach requires no fine-tuning while providing strong classification performance. These examples are randomly selected and saved in a .csv file, which can be used for all the models as an input, along with the prompt, and testing happens on those comments that do not belong to the examples csv file.

The implementation supports multiple language models, including:

- Qwen/Qwen-2.5-7B-Instruct
  
- mistralai/Mistral-7B-Instruct-v0.3
  
- deepseek-ai/DeepSeek-R1-Distill-Qwen-7B
  
- deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B

## Requirements

-Python 3.8+

-PyTorch 2.0+

-Transformers 4.30+

-pandas, numpy, matplotlib, seaborn

-scikit-learn

-CUDA-compatible GPU

## Dataset Format

The code expects CSV files with the following structure:

- Example dataset (balanced_dataset_200_comments.csv): Contains 200 examples with balanced classes (100 regional bias, 100 non-regional bias)

- Test dataset (annotated_dataset.csv): Contains comments to be classified

## Output

The code generates several outputs:

1. logs/*.log: Detailed logs of the execution
   
2. results/*/checkpoints/: Checkpoint files saved during execution
   
3. results/*_predictions_*.csv: CSV file with predictions
   
4. results/*_report_*.txt: Classification report with precision, recall, F1
   
5. results/visualizations/*_confusion_matrix_*.png: Confusion matrix visualization
   
6. results/visualizations/*_results_summary_*.png: Summary of classification results

## Security Notes

Instead of passing the HuggingFace token as a command-line argument, set it as an environment variable:

bashexport HF_TOKEN="your_token_here"
