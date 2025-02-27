import os
import pandas as pd
import anthropic

# ✅ Set Your Claude API Key
CLAUDE_API_KEY = "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"  # Replace with your actual API key

# ✅ Load Dataset
file_path = "/home/debasmita/bs_thesis/data/classified_comments_threshold_point_4.csv"
df = pd.read_csv(file_path)

# ✅ Clean Text Function
import re
def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+", "", text)  # Remove URLs
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)  # Remove special characters
    text = re.sub(r"\s+", " ", text).strip()  # Remove extra spaces
    return text

df["cleaned_text"] = df["Comment"].astype(str).apply(clean_text)

# ✅ Define Claude API Function for Sentiment Analysis
def classify_sentiment(text):
    try:
        # ✅ Set up Claude client
        client = anthropic.Anthropic(api_key=CLAUDE_API_KEY)

        # ✅ Claude prompt
        prompt = f"Classify the sentiment of the following comment into Positive, Negative, or Others:\n\n{text}\n\nSentiment:"
        
        # ✅ Make API request
        response = client.completions.create(
            model="claude-2",  # Use "claude-3" if available
            max_tokens=10,
            prompt=prompt
        )

        # ✅ Extract Sentiment from Claude's Response
        sentiment = response.completion.lower()

        if "positive" in sentiment:
            return "Positive"
        elif "negative" in sentiment:
            return "Negative"
        else:
            return "Others"

    except Exception as e:
        return "Others"

# ✅ Apply Sentiment Classification
df["sentiment_category"] = df["cleaned_text"].apply(classify_sentiment)

# ✅ Save Output
output_path = "/home/debasmita/bs_thesis/data/classified_comments_with_sentiment_claude.csv"
df.to_csv(output_path, index=False)

# ✅ Display Output
from IPython.display import display
display(df)
