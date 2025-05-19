## Regional Bias Detection - 150 Few-Shot Examples

This contains code for detecting regional biases in text comments using state-of-the-art language models with a 150-example few-shot learning approach.

## Overview

The code implements a few-shot learning pipeline where 150 annotated examples (75 with regional bias, 75 without) are used to prompt large language models to classify new comments. This approach requires no fine-tuning while providing strong classification performance. These examples are randomly selected and saved in a .csv file, which can be used for all the models as an input, along with the prompt, and testing happens on those comments that do not belong to the examples csv file.

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

- Example dataset (150_examples_few_shot_classification_dataset.csv): Contains 150 examples with balanced classes (75 regional bias, 75 non-regional bias)

- Test dataset (annotated_dataset.csv): Contains comments to be classified

## Output

The code generates several outputs:

1. logs/*.log: Detailed logs of the execution

2. results/*/checkpoints/: Checkpoint files saved during execution

3. results/predictions.csv: CSV file with predictions

4. results/report.txt: Classification report with precision, recall, F1

5. results/visualizations/confusion_matrix.png: Confusion matrix visualisation

6. results/visualizations/results_summary.png: Summary of classification results

## Security Notes

Instead of passing the HuggingFace token as a command-line argument, set it as an environment variable:

bashexport HF_TOKEN="your_token_here"
