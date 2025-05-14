# Regional-Bias-Detection-for-Indian-States-using-Large-Language-Models
## Overview

This project investigates regional social biases in India by leveraging natural language processing and large language models (LLMs). We curated a comprehensive dataset of comments from Reddit, YouTube, and Quora that contain explicit or implicit regional biases toward different Indian states and regions. Through rigorous annotation, classification, and model evaluation, we aim to understand how these biases manifest online and how effectively LLMs can identify them.

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

**3. Multi-level Annotation**

Level-1: Classify comments as regional bias and non-regional bias.

Level-2: Rate the severity of stereotypical comments (mild, moderate, severe)

Level-3: Categorize by bias types:

- Linguistic Bias
- Cultural Bias
- Economic Bias
- Political Bias
- Infrastructure Bias


Associate each stereotypical comment with specific target states/regions

**4. LLM Classification Pipeline**

Zero-shot Classification:

Evaluate multiple LLMs' ability to classify biases without examples.Test different prompting strategies to optimize zero-shot performance.

Few-shot Classification:

Design effective examples for in-context learning and test varying numbers of examples to determine an optimal few-shot configuration.

Fine-tuning Approach:

Train models on the annotated dataset. Implement cross-validation to ensure robustness. Compare performance across model sizes and architectures.

**5. Performance Evaluation**

Create benchmark metrics to evaluate LLM performance:

Accuracy, Precision, Recall, F1-score

Analyze performance differences across:

- Different Indian states/regions
- Different bias types
- Different severity levels

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
  
## Collected Data

- [Dataset (Google Drive)](https://drive.google.com/drive/folders/1uS5B-y4OAZvb9xHRS7ZrXh5QVyHtik41?usp=drive_link)

This link contains a file called final_merged_comments that has comments crawled from three different social media platforms that are Reddit, Quora and YouTube. Along with that, another file named clean_comments has the comments after cleaning the raw data, and the data cleaning methods applied here are elementary, like the removal of missing values, and empty rows, and the removal of repetitive spam comments and emojis.
  
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

- Non-Regional Bias (NRB): Comments that do not express any social bias. (Score: 0)

- Regional Bias (RB): Comments reinforcing or propagating social biases about any state based on their languages, caste, culture, economic, and infrastructural development. (Score: 1)

**Level-2: Severity of the Comments**

- Mild (Score 1): Slight bias, indirectly reinforced stereotype.

- Moderate (Score 2): Clear bias, reinforcing stereotypes with some intent.

- Severe (Score 3): Strong bias, explicitly promoting discrimination or prejudice.

**Level 3: Bias Category**

- Linguistic Bias (LB): Based on language superiority/inferiority.

- Cultural Bias (CB): Based on customs, morals, traditions, festivals, food, and lifestyle.

- Economic Bias (EB): Based on the economic status of the state, GDP per capita, financial status, and social class.

- Political Bias (PB): The political landscape, ideology, and the effect of politics in shaping people's opinions.

- Infrastructure Bias (IB): Based on urbanisation, development, public facilities, or regional accessibility. Including the forest cover and rural areas present in the state or region.

**Level 4: State/Region Assignment**

The stereotypical comments are assigned to a specific region/state that has been targeted.

The annotation process includes two groups. Disagreements were resolved through discussions to reach a consensus. We calculated inter-annotator agreement to ensure the reliability of the annotations.

## Dataset

- [Data](https://drive.google.com/drive/folders/1PISKGiWoKh-D9UPyaMCnirjOaIf2JlI7)

**Level-1**

![Screenshot from 2025-05-13 10-45-09](https://github.com/user-attachments/assets/3bd68850-7c57-479c-8c52-aae4fa454554)

This chart shows the distribution between non-biased comments (7,521) and comments containing regional bias (2,479). The visualization demonstrates that about a quarter of the analyzed comments show some form of regional bias.

**Level-2**

![Screenshot from 2025-05-13 10-45-19](https://github.com/user-attachments/assets/2ccb3686-8eff-49ad-87e2-e3dc711f51c5)

Among biased comments, moderate severity (2) is most prevalent, followed by mild (1), with severe biases (3) being the least common but still significant. This chart visualizes the severity distribution of the comments. The data shows that moderate severity comments (1,367) are most common, followed by mild (884), with severe comments (223) being less frequent but still notable.

**Level-3**

![Screenshot from 2025-05-13 10-46-12](https://github.com/user-attachments/assets/4302b6b8-7fe9-44cd-97f2-a463a431db41)

When bias is present, it most commonly manifests as a single bias type, with declining frequency as the number of simultaneous bias types increases. This visualization shows how many different types of biases appear in comments. Most biased comments contain just one type of bias (1,883), followed by two types (368), three types (156), and four types (60)

**Level-4**

![Screenshot from 2025-05-13 10-30-42](https://github.com/user-attachments/assets/42c23590-8c85-4997-84a1-63acc2621e0b)

The visualisation shows that North India has the highest number of comments (244), followed by Chhattisgarh (183), Bihar (169), and Karnataka (167). The chart displays the top 15 states/regions in descending order, making it easy to understand the distribution of comments across different Indian states and regions.

## Prompting and Classification Strategies

In the evaluation of the model's understanding of regional biases, we utilised the prompting techniques. This involved the careful curation of instructions for specific tasks into prompts that include context, definitions, and examples. We explored different prompting paradigms:

**Zero-Shot Classification:**

A classification technique that refers to the ability of large language models to perform classification solely based on the instructions and class descriptions provided in the prompt, without any seen examples of the specific task, unlike the fine-tuning technique.


- [LLaMa_3.2_3B](https://github.com/debby123p/Summarizing-personalised-social-biases-towards-Indian-states-using-different-LLMs/blob/main/zero_shot_models/llama_3.2_3b)
- [Deepseek_R1_Distill_Qwen_1.5B](https://github.com/debby123p/Summarizing-personalised-social-biases-towards-Indian-states-using-different-LLMs/blob/main/zero_shot_models/deepseek_r1_distill_qwen_7b.py)
- [Gemma_3_4B_it]
- [Mistral_7B_instruct_v0.3](https://github.com/debby123p/Summarizing-personalised-social-biases-towards-Indian-states-using-different-LLMs/blob/main/zero_shot_models/mistral_7b_instruct_v0.3)
- [Qwen_2.5_7B_instruct]
- [Deepseek_R1_Distill_Qwen_7B]

**Few-Shot Classification:**

The classification techniques involve the use of a smaller number of examples within the prompt itself. These examples illustrate the desired input-output behaviour for the task. These examples help the model better adapt its pre-trained knowledge to specific nuances in the classification. 

**Few-Shot (Support-100)**

The examples are randomly selected from the annotated dataset, that is, **50 comments tagged regional biases and 50 comments tagged as non-regional biases**, as input to the prompt as support examples, helping the model to understand the nuances better.

- [Deepseek_R1_Distill_Qwen_1.5B](https://github.com/debby123p/Summarizing-personalised-social-biases-towards-Indian-states-using-different-LLMs/blob/main/few_shot_100_examples/deepseek_r1_distill_qwen_1.5b.py)
- [Mistral_7B_instruct_v0.3](https://github.com/debby123p/Summarizing-personalised-social-biases-towards-Indian-states-using-different-LLMs/blob/main/few_shot_100_examples/mistral_7b_instruct_v0.3.py)
- [Qwen_2.5_7B_instruct](https://github.com/debby123p/Summarizing-personalised-social-biases-towards-Indian-states-using-different-LLMs/blob/main/few_shot_100_examples/qwen_2.5_7b_instruct.py)
- [Deepseek_R1_Distill_Qwen_7B]

**Few-Shot (Support-150)**

The examples are randomly selected from the annotated dataset, that is, **75 comments tagged regional biases and 75 comments tagged as non-regional biases**, as input to the prompt as support examples, helping the model to understand the nuances better.

- [Deepseek_R1_Distill_Qwen_1.5B]
- [Mistral_7B_instruct_v0.3](https://github.com/debby123p/Regional-Bias-Detection-for-Indian-States-using-Large-Language-Models/blob/main/few_shot_150_examples/Mistral_7b_instruct_v0.3.py)
- [Qwen_2.5_7B_instruct](https://github.com/debby123p/Regional-Bias-Detection-for-Indian-States-using-Large-Language-Models/blob/main/few_shot_150_examples/qwen_2.5_7b_instruct.py)
- [Deepseek_R1_Distill_Qwen_7B]
  
**Few-Shot (Support-200)**

The examples are randomly selected from the annotated dataset, that is, **100 comments tagged regional biases and 100 comments tagged as non-regional biases**, as input to the prompt as support examples, helping the model to understand the nuances better.

- [Deepseek_R1_Distill_Qwen_1.5B](https://github.com/debby123p/Regional-Bias-Detection-for-Indian-States-using-Large-Language-Models/blob/main/few_shot_200_examples/Deepseek_r1_distill_1.5b.py)
- [Mistral_7B_instruct_v0.3](https://github.com/debby123p/Regional-Bias-Detection-for-Indian-States-using-Large-Language-Models/blob/main/few_shot_200_examples/Mistral_7b_instruct_v0.3.py)
- [Qwen_2.5_7B_instruct](https://github.com/debby123p/Regional-Bias-Detection-for-Indian-States-using-Large-Language-Models/blob/main/few_shot_200_examples/Qwen_2.5_7b_instruct.py)
- [Deepseek_R1_Distill_Qwen_7B](https://github.com/debby123p/Regional-Bias-Detection-for-Indian-States-using-Large-Language-Models/blob/main/few_shot_200_examples/Deepseek_r1_distill_7b.py)





    



