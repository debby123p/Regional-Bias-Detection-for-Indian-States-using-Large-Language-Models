# Summarizing-personalised-social-biases-towards-Indian-states-using-LLMs
## Overview

This project aims to analyze and summarize state-wise social biases in India using large language models (LLMs). Leveraging Reddit, Quora, and YouTube data, we extract and analyze public sentiments, classify them, and summarize biases using different LLMs. The goal is to create a dataset that captures state-specific biases and evaluate various LLMs' ability to summarize these biases accurately.

## Project Workflow

**1. Dataset Preparation:**
   
Crawl and extract state-wise social bias-related comments from:

         a) Reddit

         b) Quora

         c) YouTube

Preprocess and clean the dataset to remove noise and irrelevant data.

**2.  Sentiment Classification:**

Apply a sentiment classifier to each comment to assign a sentiment score.

Categorize comments into positive and negative classes.

**3.  Grouping by State:**

Aggregate comments based on states.

Maintain sentiment scores to retain polarity information.

**4. LLM-Based Summarization:**

Feed grouped state-wise comments into different LLMs.

Summarize the positive and negative biases about each state.

Compare model performance using summarization scores.

**5. Fine-Tuning LLMs:**

Use the gathered dataset to fine-tune LLMs.

Improve the models’ understanding of deep-rooted social biases in India.

**6. Future Prospects:**

Develop a psychometric test to detect unconscious social biases in hiring policies.

Use LLMs to assess personal biases and their impact on decision-making.

Train LLMs to recognize and counteract subtle biases in language processing.

## Literature Review

Here are some papers I've studied for this project:

- *IndiBias: A Benchmark Dataset to Measure Social Biases in Language Models for Indian Context*- [Read here](https://arxiv.org/abs/2403.20147)
- *MuRIL: Multilingual Representations for Indian Languages* - [Read here](https://arxiv.org/abs/2103.10730)
- *Socially Aware Bias Measurements for Hindi Language Representations* - [Read here](https://arxiv.org/abs/2110.07871)
- *StereoSet: Measuring stereotypical bias in pretrained language models* - [Read Here](https://aclanthology.org/2021.acl-long.416/)
- *Differences in decisions affected by cognitive biases: examining human values, need for cognition, and numeracy* - [Read Here](https://prc.springeropen.com/articles/10.1186/s41155-023-00265-z?utm_source=chatgpt.com)
- *Behavioural Biases and Investment Decision-Making in India:A Study of Stock Market Investors* - [Read Here](https://irjems.org/Volume-3-Issue-11/IRJEMS-V3I11P102.pdf)

## Studying regional biases
- [List of regional biases](https://docs.google.com/spreadsheets/d/1eKwjUe5UhHZSvtNiS4Kwy0dyowl7G2V6oPXfzHFRATg/edit?usp=sharing)

This is a list of regional biases in India, categorized into various types of cognitive biases. The information is gathered through an extensive study of news articles and social media discussions, as well as a Google Form circulated among my classmates.
  
## Dataset 

- [Dataset (Google Drive)](https://drive.google.com/drive/folders/1uS5B-y4OAZvb9xHRS7ZrXh5QVyHtik41?usp=drive_link)

This link contains a file called final_merged_comments that has comments crawled from three different social media platforms that are Reddit, Quora and YouTube. Along with that, another file named clean_comments has the comments after cleaning the raw data, and the data cleaning methods applied here are elementary, like the removal of missing values, and empty rows, and the removal of repetitive spam comments and emojis.


## Presentations 

- [Presentation-1](https://docs.google.com/presentation/d/1FpWwApohY7X4-R5gs5h47mRoa3VcavL6whEYGOiM4mQ/edit?usp=sharing)
  
This presentation has information about different biases existing in the country in the context of regional stereotypes, myths and beliefs. Along with a breakdown of state-wise biases in axes like caste, regionality, occupation, physical looks and habits( food, alcohol, tobacco, loud, fun, etc).
  
- [Presentation-2](https://docs.google.com/presentation/d/1B6vRR1Crb4xsdH-tQWH9BkGmoPiLYEBEb1AikXks-Vs/edit?usp=sharing)

This presentation discusses a paper on the IndiBias dataset, its preparation, and usage with LLM models. We also talked about the policies we will follow while preparing our dataset.

- [Presentation-3](https://docs.google.com/presentation/d/17diW7yOIRtt_v0C7wlCgANla-Mhv1QYeCot7fvIYDSQ/edit?usp=sharing)

In this presentation, we discussed the initial crawled data from Reddit and how we can utilize this data for our study, along with a discussion on increasing the dataset.

- [Presentation-4](https://docs.google.com/presentation/d/1EOsByCeHv7QQt2Xl9uS-QlDkeVRkMvMI0lCIEopzjMA/edit?usp=sharing)

In this presentation, we clearly outline the data collected from various social media platforms and detail how we constructed the dataset by merging all comments and responses. We also emphasize our comprehensive data cleaning process, which involved addressing missing values, eliminating duplicates, normalizing text, and removing repetitive comments.

- [Presentation(Mid-sem Evaluation)](https://docs.google.com/presentation/d/14-uCZWOnULY_gTC-6fnmfTM6U07NvvCodDSRbGkF2rc/edit?usp=sharing)

  
## Code

- [Data_cleaning](https://colab.research.google.com/drive/1qEGHVvUY9JrtbDrsh5AwGrvR2eSdiP5T?authuser=0#scrollTo=Bam4mgJi-HXW)

This file contains the cleaning of the dataset, where primary cleaning is done, along with utilising the SBERT model, we have calculated the word-similarity score to categorise comments in the following categories:

1) Positive
2) Negative
3) Others
