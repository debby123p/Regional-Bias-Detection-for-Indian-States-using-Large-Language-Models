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

- [Standard Sbert model](https://github.com/debby123p/Summarizing-personalised-social-biases-towards-Indian-states-using-different-LLMs/blob/main/standard_sbert.py)
- [MiniLM_L6_model](https://github.com/debby123p/Summarizing-personalised-social-biases-towards-Indian-states-using-different-LLMs/blob/main/minilm_l6_sbert.py)
  
  This is for filtering comments relevant to my project using the SBERT model, here we have once used the standard SBERT model and MiniLM_L6(Distilled SBERT), which are the above two models.

  ## Sentiment Classification

- [RoBERTa](https://github.com/debby123p/Summarizing-personalised-social-biases-towards-Indian-states-using-different-LLMs/blob/main/sentiment_classification_roberta.py)
- [mBERT](https://github.com/debby123p/Summarizing-personalised-social-biases-towards-Indian-states-using-different-LLMs/blob/main/sentiment_classification_mbert.py)
- [MURIL](https://github.com/debby123p/Summarizing-personalised-social-biases-towards-Indian-states-using-different-LLMs/blob/main/sentiment_classification_muril.py)
- [IndicBERT](https://github.com/debby123p/Summarizing-personalised-social-biases-towards-Indian-states-using-different-LLMs/blob/main/sentiment_classification_indicbert.py)
- [XLM_RoBERTa](https://github.com/debby123p/Summarizing-personalised-social-biases-towards-Indian-states-using-different-LLMs/blob/main/sentiment_classification_xlm_roberta.py)
- [XLM_RoBERTa_with_thresholding](https://github.com/debby123p/Summarizing-personalised-social-biases-towards-Indian-states-using-different-LLMs/blob/main/sentiment_classification_xlm_roberta_0.5.py)
- [LLaMa_3.2_1B](https://github.com/debby123p/Summarizing-personalised-social-biases-towards-Indian-states-using-different-LLMs/blob/main/sentiment_classification_llama_3.2_1b.py)
- [LLaMa_3.2_1B_instruct](https://github.com/debby123p/Summarizing-personalised-social-biases-towards-Indian-states-using-different-LLMs/blob/main/sentiment_classification_llama_3.2_1b_instruct.py)
- [LLaMa_3.2_3B](https://github.com/debby123p/Summarizing-personalised-social-biases-towards-Indian-states-using-different-LLMs/blob/main/sentiment_classification_llama_3.2_3b.py)
- [Claude](https://github.com/debby123p/Summarizing-personalised-social-biases-towards-Indian-states-using-different-LLMs/blob/main/sentiment_classification_claude.py)
- [Mistral_7B](https://github.com/debby123p/Summarizing-personalised-social-biases-towards-Indian-states-using-different-LLMs/blob/main/sentiment_classification_mistral_7B.py)

## Dataset Annotation 

We have developed a comprehensive annotation framework to understand the nature and intensity of the regional biases. 

**Level 1: Regional Bias vs Non-Regional Bias**

Non-Regional Bias (NRB): Comments that do not express any social bias. (Score: 0)

Regional Bias (RB): Comments reinforcing or propagating social biases about any state based on their languages, caste, culture, economic, and infrastructural development. (Score: 1)

**Level-2: Severity of the Comments**

Mild (Score 1): Slight bias, indirectly reinforced stereotype.

Moderate (Score 2): Clear bias, reinforcing stereotypes with some intent.

Severe (Score 3): Strong bias, explicitly promoting discrimination or prejudice.

**Level 3: Bias Category**

Linguistic Bias (LB): Based on language superiority/inferiority.

Cultural Bias (CB): Based on customs, morals, traditions, festivals, food, and lifestyle.

Economic Bias (EB): Based on the economic status of the state, GDP per capita, financial status, and social class.

Political Bias (PB): The political landscape, ideology, and the effect of politics in shaping people's opinions.

Infrastructure Bias (IB): Based on urbanisation, development, public facilities, or regional accessibility. Including the forest cover and rural areas present in the state or region.

**Level 4: State/Region Assignment**

The stereotypical comments are assigned to a specific region/state that has been targeted.

The annotation process includes two groups. Disagreements were resolved through discussions to reach a consensus. We calculated inter-annotator agreement to ensure the reliability of the annotations.

## Dataset

-[Data](https://drive.google.com/drive/folders/1PISKGiWoKh-D9UPyaMCnirjOaIf2JlI7)


## Prompting and Classification Strategies

In the evaluation of the model's understanding of regional biases, we utilised the prompting techniques. This involved the careful curation of instructions for specific tasks into prompts that include context, definitions, and examples. We explored different prompting paradigms:

**Zero-Shot Classification:**

A classification technique that refers to the ability of large language models to perform classification solely based on the instructions and class descriptions provided in the prompt, without any seen examples of the specific task, unlike the fine-tuning technique.


- [LLaMa_3.2_3B](https://github.com/debby123p/Summarizing-personalised-social-biases-towards-Indian-states-using-different-LLMs/blob/main/zero_shot_models/llama_3.2_3b)
- [Deepseek_R1_Distill_Qwen_1.5B](https://github.com/debby123p/Summarizing-personalised-social-biases-towards-Indian-states-using-different-LLMs/blob/main/zero_shot_models/deepseek_r1_distill_qwen_7b.py)
- [Mistral_7B_instruct_v0.3](https://github.com/debby123p/Summarizing-personalised-social-biases-towards-Indian-states-using-different-LLMs/blob/main/zero_shot_models/mistral_7b_instruct_v0.3)

**Few-Shot Classification:**

The classification techniques involve the use of a smaller number of examples within the prompt itself. These examples illustrate the desired input-output behaviour for the task. These examples help the model better adapt its pre-trained knowledge to specific nuances in the classification. 

The examples are randomly selected from the annotated dataset, that is, **50 comments tagged regional biases and 50 comments tagged as non-regional biases**, as input to the prompt as support examples, helping the model to understand the nuances better.

**Few-Shot (Support-100)**

-[Deepseek_R1_Distill_Qwen_1.5B](https://github.com/debby123p/Summarizing-personalised-social-biases-towards-Indian-states-using-different-LLMs/blob/main/few_shot_100_examples/deepseek_r1_distill_qwen_1.5b.py)
-[Mistral_7B_instruct_v0.3](https://github.com/debby123p/Summarizing-personalised-social-biases-towards-Indian-states-using-different-LLMs/blob/main/few_shot_100_examples/mistral_7b_instruct_v0.3.py)






    



