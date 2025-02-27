import pandas as pd
import re
import nltk
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from nltk.corpus import stopwords
import emoji

# Download stopwords if not already downloaded
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# Additional stopwords for Indian languages
indian_stopwords = set(["hai", "ka", "ke", "ki", "hota", "nahi", "kyu", "kya", "ho", "kar", "jo", "toh", "se"])
stop_words.update(indian_stopwords)

# Load the dataset
file_path = "/home/debasmita/bs_thesis/data/distilled_sbert_classified_comments_threshold_point_04.csv"
df = pd.read_csv(file_path)

# Remove "Uncertain" category comments
df_filtered = df[df['Bias_Category'] != 'Uncertain']

# Remove empty or very short comments (less than 3 characters)
df_filtered = df_filtered[df_filtered['Comment'].astype(str).str.len() > 3]

# Function to clean text: Remove special characters, stopwords, emojis, and repetitive words
def clean_text(text):
    text = str(text).lower()  # Lowercasing
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)  # Remove special characters
    text = emoji.replace_emoji(text, replace='')  # Remove emojis
    words = text.split()
    words = [word for word in words if word not in stop_words]  # Remove stopwords
    text = ' '.join(sorted(set(words), key=words.index))  # Remove repetitive words while keeping order
    return text.strip()

df_filtered['Comment'] = df_filtered['Comment'].apply(clean_text)

# Load XLM-RoBERTa Model for Sentiment Analysis
model_name = "cardiffnlp/twitter-xlm-roberta-base-sentiment"  # Use a valid fine-tuned model
device = "cuda" if torch.cuda.is_available() else "cpu"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name).to(device)
classifier = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer, device=0 if device == "cuda" else -1)

# Function to handle long comments using sliding window
def assign_sentiment_score(comment):
    try:
        max_length = 512  # XLM-RoBERTa max token limit
        tokens = tokenizer(comment, truncation=False, return_tensors="pt")

        # If comment exceeds max length, split into chunks
        if tokens["input_ids"].shape[1] > max_length:
            chunks = [comment[i:i+max_length] for i in range(0, len(comment), max_length)]
            results = [classifier(chunk)[0] for chunk in chunks]

            # Aggregate results using weighted confidence score
            sentiment_scores = {"LABEL_0": 0, "LABEL_1": 0, "LABEL_2": 0}
            for res in results:
                sentiment_scores[res["label"]] += res["score"]

            best_sentiment = max(sentiment_scores, key=sentiment_scores.get)
            sentiment_label = best_sentiment
            sentiment_score = sentiment_scores[best_sentiment] / len(results)  # Averaging scores
        else:
            # Run classification for shorter text
            result = classifier(comment)[0]
            sentiment_label = result["label"]
            sentiment_score = result["score"]

        # Adjusted classification thresholds (Dynamic)
        if sentiment_label == "LABEL_2" and sentiment_score >= 0.5:  # Positive
            return "Positive", sentiment_score
        elif sentiment_label == "LABEL_0" and sentiment_score >= 0.5:  # Negative
            return "Negative", sentiment_score
        else:  # Neutral -> Mapped to best-fit category
            if sentiment_score > 0.45:
                return "Positive", sentiment_score
            elif sentiment_score < 0.45:
                return "Negative", sentiment_score
            else:
                return "Other", sentiment_score

    except Exception as e:
        return "Other", 0.0  # Return 'Other' with score 0 if there's an error

# Apply sentiment classification to dataset
df_filtered['Sentiment_Category'], df_filtered['Sentiment_Score'] = zip(*df_filtered['Comment'].apply(assign_sentiment_score))

# Save the updated dataset with sentiment scores
output_file_path = "/home/debasmita/bs_thesis/data/classified_comments_with_fixed_xlm_roberta_sentiment.csv"
df_filtered.to_csv(output_file_path, index=False)

# Display summary of classified comments
print("\nSentiment Category Distribution:\n", df_filtered['Sentiment_Category'].value_counts())
print(f"Updated dataset saved at: {output_file_path}")




