import os
import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, BitsAndBytesConfig
import torch.nn as nn

#  Prevent Deadlocks & Optimize CUDA
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"

#  Load Dataset
file_path = "/home/debasmita/bs_thesis/data/distillied_sbert_classified_comments_threshold_point_04.csv"
df = pd.read_csv(file_path)

#  Clean Text Function
import re
def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+", "", text)  # Remove URLs
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)  # Remove special characters
    text = re.sub(r"\s+", " ", text).strip()  # Remove extra spaces
    return text

df["cleaned_text"] = df["Comment"].astype(str).apply(clean_text)

#  Load LLaMA Model (Optimized for 8GB VRAM)
model_name = "meta-llama/Llama-3.2-1B"

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,  
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16
)

tokenizer = AutoTokenizer.from_pretrained(model_name)

model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    quantization_config=quantization_config,
    device_map="auto",
    num_labels=3  # Ensure model is set for classification
)

# Add classification head (Fix missing `score.weight` issue)
if not hasattr(model, "classifier"):
    model.classifier = nn.Linear(model.config.hidden_size, 3)  # ✅ Define classification layer

#  Ensure Model is in Evaluation Mode
model.eval()

#  Function to Classify Sentiment
def classify_sentiment(text):
    try:
        # Tokenize input with max 512 token limit
        tokens = tokenizer(text, truncation=True, max_length=512, return_tensors="pt").to("cuda")

        # Get model output
        with torch.no_grad():
            outputs = model(**tokens)
        
        #  Extract Logits & Apply Softmax
        probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
        scores = probabilities.cpu().numpy()[0]

        # Get Predicted Sentiment Category
        sentiment_labels = ["Negative", "Other", "Positive"]
        sentiment_category = sentiment_labels[scores.argmax()]
        
        return scores.max(), sentiment_category  # ✅ Return confidence score & category
    
    except Exception as e:
        return 0, "Others"

# Apply Sentiment Classification
df[["sentiment_score", "sentiment_category"]] = df["cleaned_text"].apply(lambda x: pd.Series(classify_sentiment(x)))

#  Save Output
output_path = "/home/debasmita/bs_thesis/data/classified_comments_with_sentiment_llama.csv"
df.to_csv(output_path, index=False)

#  Display Output
from IPython.display import display
display(df)

