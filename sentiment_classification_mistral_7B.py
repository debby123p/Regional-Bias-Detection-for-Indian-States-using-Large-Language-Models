import torch
import gc
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

# Free up GPU memory before loading
torch.cuda.empty_cache()
gc.collect()

# Define model name
model_name = "mistralai/Mistral-7B-Instruct-v0.3"

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Define optimized quantization config
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,  # Use 4-bit quantization to reduce memory
    bnb_4bit_compute_dtype=torch.float16,  # Use FP16 for efficiency
    bnb_4bit_use_double_quant=True,  # Double quantization for stability
    bnb_4bit_quant_type="nf4",  # More efficient quantization type
)

# Load Mistral-7B model with quantization
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,  # Correct quantization format
    device_map="auto",  # Automatically place model on GPU
    torch_dtype=torch.float16,  # FP16 to save memory
)

print("✅ Mistral-7B successfully loaded!")


import pandas as pd
import re
import emoji
import nltk
from nltk.corpus import stopwords

# Download NLTK stopwords
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# Additional stopwords for Indian languages
indian_stopwords = {"hai", "ka", "ke", "ki", "hota", "nahi", "kyu", "kya", "ho", "kar", "jo", "toh", "se"}
stop_words.update(indian_stopwords)

# Load dataset
file_path = "/home/debasmita/bs_thesis/data/distilled_sbert_classified_comments_threshold_point_04.csv"  # Update with your actual path
df = pd.read_csv(file_path)

# Remove "Uncertain" category comments
df_filtered = df[df['Bias_Category'] != 'Uncertain']

# Remove empty or very short comments (less than 3 characters)
df_filtered = df_filtered[df_filtered['Comment'].astype(str).str.len() > 3]

# Function to clean text
def clean_text(text):
    text = str(text).lower()  # Lowercasing
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)  # Remove special characters
    text = emoji.replace_emoji(text, replace='')  # Remove emojis
    words = text.split()
    words = [word for word in words if word not in stop_words]  # Remove stopwords
    text = ' '.join(sorted(set(words), key=words.index))  # Remove repetitive words while keeping order
    return text.strip()

# Apply cleaning
df_filtered['Cleaned_Comment'] = df_filtered['Comment'].apply(clean_text)
print("✅ Dataset loaded and preprocessed!")


# Function to classify sentiment using Mistral-7B
def classify_sentiment(comment):
    try:
        # Create prompt for sentiment analysis
        prompt = f"Analyze the sentiment of this comment: '{comment}' and classify it as Positive, Negative, or Other."

        # Tokenize input text
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

        # Generate output
        output = model.generate(**inputs, max_length=150)
        decoded_output = tokenizer.decode(output[0], skip_special_tokens=True)

        # Extract sentiment label from response
        if "Positive" in decoded_output:
            return "Positive"
        elif "Negative" in decoded_output:
            return "Negative"
        else:
            return "Other"

    except Exception as e:
        print(f"Error processing comment: {comment} - {e}")
        return "Other"  # Default to "Other" in case of an error

# Apply sentiment classification to dataset
df_filtered['Sentiment_Category'] = df_filtered['Cleaned_Comment'].apply(classify_sentiment)

# Save the updated dataset with sentiment labels
output_file_path = "/home/debasmita/bs_thesis/data/classified_comments_with_mistral7b.csv"
df_filtered.to_csv(output_file_path, index=False)

# Display summary of classified comments
print("\n✅ Sentiment Classification Completed!")
print(df_filtered['Sentiment_Category'].value_counts())
print(f"Updated dataset saved at: {output_file_path}")
