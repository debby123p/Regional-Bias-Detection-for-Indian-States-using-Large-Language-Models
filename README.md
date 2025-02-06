# Summarizing-personalised-social-biases-towards-Indian-states-using-LLMs
**Overview:**

This project aims to analyze and summarize state-wise social biases in India using large language models (LLMs). Leveraging Reddit, Quora, and YouTube data, we extract and analyze public sentiments, classify them, and summarize biases using different LLMs. The goal is to create a dataset that captures state-specific biases and evaluate various LLMs' ability to summarize these biases accurately.

**Project Workflow:**

**1. Dataset Preparation**
   
Crawl and extract state-wise social bias-related comments from:

         a) Reddit

         b) Quora

         c) YouTube

Preprocess and clean the dataset to remove noise and irrelevant data.

**2.  Sentiment Classification**

Apply a sentiment classifier to each comment to assign a sentiment score.

Categorize comments into positive and negative classes.

**3.  Grouping by State**

Aggregate comments based on states.

Maintain sentiment scores to retain polarity information.

**4. LLM-Based Summarization**

Feed grouped state-wise comments into different LLMs.

Summarize the positive and negative biases about each state.

Compare model performance using summarization scores.

**5. Fine-Tuning LLMs**

Use the gathered dataset to fine-tune LLMs.

Improve the models’ understanding of deep-rooted social biases in India.

**6. Future Prospects**

Develop a psychometric test to detect unconscious social biases in hiring policies.

Use LLMs to assess personal biases and their impact on decision-making.

Train LLMs to recognize and counteract subtle biases in language processing.


## Dataset 

- [Dataset (Google Drive)](https://drive.google.com/drive/folders/1uS5B-y4OAZvb9xHRS7ZrXh5QVyHtik41?usp=drive_link)

This link contains a file called final_merged_comments that has comments crawled from three different social media platforms that are Reddit, Quora and YouTube. Along with that, there is another file named clean_comments that has the comments after cleaning the raw data, and the data cleaning methods applied here are very simple, like removal of missing values, empty rows, and removal of repetitive spam comments and emojis.



