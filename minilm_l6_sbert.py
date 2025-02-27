import pandas as pd
import torch
from sentence_transformers import SentenceTransformer, util

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# ✅ Load the dataset
file_path = "/home/debasmita/bs_thesis/data/clean_comments.csv"  
df = pd.read_csv(file_path)

# ✅ Ensure all comments are strings and remove NaN values
df = df.dropna(subset=['Comment'])
df['Comment'] = df['Comment'].astype(str)

# ✅ Load the Distilled SBERT Model (MiniLM-L6)
model = SentenceTransformer('all-MiniLM-L6-v2', device=device)  # Distilled SBERT Model

# ✅ Define Predefined Keywords for Bias Categories
positive_keywords = [
    "culture", "love", "respect", "study", "smart", "intelligent", "Hindi", "region", "known",
    "new", "true", "less", "rest", "home", "beautiful", "actually", "big", "whole", "cultural",
    "help", "community", "kind", "job", "having", "believe", "ask", "friend", "high", "biggest",
    "development", "long", "especially", "famous", "educated", "often", "cities", "learning",
    "keep", "school", "central", "major", "start", "getting", "point", "developed", "capital",
    "majority", "nation", "everything", "group", "quite", "fun", "happy", "largest", "sweet",
    "similar", "enough", "highest", "baat", "communicate", "knowledge", "support", "consider",
    "locals", "diversity", "heritage", "progressive", "hardworking", "peaceful", "friendly",
    "creative", "care", "bharat", "several", "popular", "traditional", "unique", "strong", "heart",
    "together", "literacy", "unity", "desh", "economic", "funny", "beauty", "freedom",
    "independence", "safe", "comfortable", "cool", "peace",
    "success", "happiness", "growth", "respectful", "unity", "progress", "hardworking",
    "leader", "amazing", "brilliant", "support", "innovative", "friendly", "cooperation",
    "achievement", "strong", "prosperity", "creative", "trust", "historic", "together",
    "visionary", "motivated", "kindness", "love", "independence", "encouragement",
    "celebration", "proud", "diverse", "optimistic", "flourish", "beautiful", "literacy",
    "economic", "freedom", "opportunity", "tradition", "bright", "inspiration", "smile",
    "safe", "famous", "talented", "joy", "popular", "brave", "courageous", "respectful",
    "hospitality", "educational", "ambition", "harmony", "polite", "healthy", "grateful", "army"
]

negative_keywords = [
    "Hindi", "south", "Bihari", "Bengali", "kannada", "kanndigas", "Bangalore", "Biharis",
    "gujjus", "Gujarati", "Punjabi", "loud", "alcohol", "drugs", "alcoholism", "racism",
    "bias", "stereotypes", "non-veg", "marathi", "Karnataka", "Tamilians", "north",
    "Indians", "hai", "tamil", "states", "Bihar", "West Bengal", "telugu", "english",
    "karnatak", "delhi", "country", "hate", "kerala", "national", "pradesh", "marathi",
    "problem", "west", "bengalis", "problem", "population", "odisha", "bad", "region",
    "education", "telangana", "reason", "against", "money", "mumbai", "name", "assam",
    "between", "without", "rich", "main", "hain", "does", "saying", "misconceptions",
    "caste", "eat", "nhi", "accent", "mean", "question", "gujarat", "bhai", "haryana",
    "non", "community", "malayalam", "hyderabad", "chhattisgarh", "northeast", "wrong",
    "chennai", "hindu", "nothing", "native", "hard", "southern", "kannadiga", "political",
    "himachal", "mein", "stop", "outside", "koi", "poor", "bollywood", "jharkhand",
    "business", "mind", "small", "kya", "old", "jai", "kolkata", "experience", "sab",
    "mai", "hote", "religion", "official", "odia", "rajasthan", "compared", "kar",
    "tamils", "issue", "hum", "social", "telugus", "stereotypes", "politicians",
    "northern", "identity", "regional", "media", "power", "rice", "difference", "regions",
    "fish", "discrimination", "companies", "case", "assamese", "end", "muslim",
    "tamilnadu", "racist", "tamilians", "backward", "lazy", "corrupt", "illiterate"
]

# Encode keywords using MiniLM-L6
positive_embeddings = model.encode(positive_keywords, convert_to_tensor=True)
negative_embeddings = model.encode(negative_keywords, convert_to_tensor=True)

# Define similarity thresholds
thresholds = [0.6, 0.5, 0.45, 0.4, 0.3, 0.1]

# Function for comment classification
def classify_comment(comment, threshold):
    """Uses MiniLM-L6 to classify comments based on keyword similarity at a given threshold."""
    comment_embedding = model.encode(comment, convert_to_tensor=True)

    # Compute similarity scores
    positive_score = util.cos_sim(comment_embedding, positive_embeddings).max().item()
    negative_score = util.cos_sim(comment_embedding, negative_embeddings).max().item()

    # Assign category based on highest similarity
    scores = {"Positive Bias": positive_score, "Negative Bias": negative_score}
    category = max(scores, key=scores.get)
    similarity_score = scores[category]

    # Apply threshold filtering
    if similarity_score < threshold:
        return "Uncertain", similarity_score

    return category, similarity_score

# Iterate over different thresholds and classify data
for threshold in thresholds:
    df_copy = df.copy()  # Make a copy to avoid modifying the original dataframe
    df_copy['Bias_Category'], df_copy['Similarity_Score'] = zip(*df_copy['Comment'].apply(lambda x: classify_comment(x, threshold)))

    # Save the classified dataset for each threshold
    output_file = f"/home/debasmita/bs_thesis/data/distilled_sbert_classified_comments_threshold_{int(threshold*100)}.csv"
    df_copy.to_csv(output_file, index=False)

    # Display Summary for each threshold
    print(f"\n--- Threshold: {threshold} ---")
    print(df_copy['Bias_Category'].value_counts())
    print(f"Classified dataset saved at: {output_file}")
