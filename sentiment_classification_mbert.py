import os
import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline

# ✅ Load Dataset
file_path = "/home/debasmita/bs_thesis/data/classified_comments_threshold_point_4.csv"
df = pd.read_csv(file_path)

# ✅ Load Pretrained mBERT Model for Sentiment Analysis
model_name = "bert-base-multilingual-uncased"  # Multilingual BERT (mBERT)
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=3)  # Assuming 3 sentiment labels

# ✅ Define Sentiment Classification Pipeline
classifier = pipeline("text-classification", model=model, tokenizer=tokenizer, device=0 if torch.cuda.is_available() else -1)

# ✅ Function to Classify Sentiment Using mBERT
def classify_sentiment(text):
    try:
        # ✅ Tokenize input with max 512 token limit
        tokens = tokenizer(text, truncation=True, max_length=512, return_tensors="pt").to(model.device)

        # ✅ Get model output
        with torch.no_grad():
            outputs = model(**tokens)
        
        # ✅ Extract Logits & Apply Softmax
        probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
        scores = probabilities.cpu().numpy()[0]

        # ✅ Map mBERT Labels to Sentiment Categories
        sentiment_labels = ["Negative", "Others", "Positive"]  # Assuming 0=Negative, 1=Neutral, 2=Positive
        sentiment_category = sentiment_labels[scores.argmax()]
        
        return sentiment_category, scores.max()  # ✅ Return sentiment category & confidence score
    
    except Exception as e:
        return "Others", 0.0

# ✅ Apply Sentiment Classification
df[["Sentiment_Category", "Sentiment_Score"]] = df["Comment"].astype(str).apply(lambda x: pd.Series(classify_sentiment(x)))

# ✅ Save Output
output_path = "/home/debasmita/bs_thesis/data/classified_comments_with_sentiment_mbert.csv"
df.to_csv(output_path, index=False)

# ✅ Print Summary
print("Sentiment Classification Completed Using mBERT.")
print(df[["Comment", "Sentiment_Category", "Sentiment_Score"]].head())
