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


## Presentations 

- [Presentation-1](https://docs.google.com/presentation/d/1FpWwApohY7X4-R5gs5h47mRoa3VcavL6whEYGOiM4mQ/edit?usp=sharing)
  
This presentation has information about different biases existing in the country in the context of regional stereotypes, myths and beliefs. Along with a breakdown of state-wise biases in axes like caste, regionality, occupation, physical looks and habits( food, alcohol, tobacco, loud, fun, etc).
  
- [Presentation-2](https://docs.google.com/presentation/d/1B6vRR1Crb4xsdH-tQWH9BkGmoPiLYEBEb1AikXks-Vs/edit?usp=sharing)

This presentation discusses a paper on the IndiBias dataset, its preparation, and usage with LLM models. We also talked about the policies we will follow while preparing our dataset.

- [Presentation-3](https://docs.google.com/presentation/d/17diW7yOIRtt_v0C7wlCgANla-Mhv1QYeCot7fvIYDSQ/edit?usp=sharing)
In this presentation, we discussed the initial crawled data from Reddit and how we can utilize this data for our study, along with a discussion on increasing the dataset.

- [Presentation-4](https://docs.google.com/presentation/d/1EOsByCeHv7QQt2Xl9uS-QlDkeVRkMvMI0lCIEopzjMA/edit?usp=sharing)

In this presentation, we clearly outline the data collected from various social media platforms and detail how we constructed the dataset by merging all comments and responses. We also emphasize our comprehensive data cleaning process, which involved addressing missing values, eliminating duplicates, normalizing text, and removing repetitive comments.
