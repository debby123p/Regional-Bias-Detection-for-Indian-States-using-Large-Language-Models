import os
import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline

# ✅ Load Dataset
file_path = "/home/debasmita/bs_thesis/data/classified_comments_threshold_point_4.csv"
df = pd.read_csv(file_path)

# ✅ Load Pretrained RoBERTa Model for Sentiment Analysis
model_name = "cardiffnlp/twitter-roberta-base-sentiment"  # Pretrained RoBERTa for sentiment analysis
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

# ✅ Define Sentiment Classification Pipeline
classifier = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer, device=0 if torch.cuda.is_available() else -1)

# ✅ Function to Classify Sentiment Using RoBERTa
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

        # ✅ Map RoBERTa Labels to Sentiment Categories
        sentiment_labels = ["Negative", "Others", "Positive"]  # RoBERTa uses: LABEL_0=Negative, LABEL_1=Neutral, LABEL_2=Positive
        sentiment_category = sentiment_labels[scores.argmax()]
        
        return sentiment_category, scores.max()  
    
    except Exception as e:
        return "Others", 0.0

# ✅ Apply Sentiment Classification
df[["Sentiment_Category", "Sentiment_Score"]] = df["Comment"].astype(str).apply(lambda x: pd.Series(classify_sentiment(x)))

# ✅ Save Output
output_path = "/home/debasmita/bs_thesis/data/classified_comments_with_sentiment_roberta.csv"
df.to_csv(output_path, index=False)

# ✅ Print Summary
print("Sentiment Classification Completed.")
print(df[["Comment", "Sentiment_Category", "Sentiment_Score"]].head())
