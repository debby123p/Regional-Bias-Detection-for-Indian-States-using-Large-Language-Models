# Detecting-Indian-Regional-Biases-in-Online-Social-Interactions-using-Large-Language-Models
## Overview

This project investigates regional social biases in India by leveraging natural language processing and large language models (LLMs). We curated a comprehensive dataset of comments from Reddit, YouTube, and Quora that contain explicit or implicit regional biases toward different Indian states and regions. Through rigorous annotation, classification, and model evaluation, we aim to understand how these biases manifest online and how effectively LLMs can identify them.

## Project Workflow

**1. Dataset Preparation:**
   
Crawl and extract state-wise social bias-related comments from:

         a) Reddit
         
         b) YouTube

Preprocess and clean the dataset to remove noise and irrelevant data.

**2.  Sentiment Classification:**

Apply a sentiment classifier to each comment to assign a sentiment score.

Categorise comments into positive and negative classes.

**3. Multi-level Annotation**

Level-1: Classify comments as regional bias and non-regional bias.

Level-2: Rate the severity of stereotypical comments (mild, moderate, severe)

Level-3: Categorise by bias types:

- Linguistic Bias
- Cultural Bias
- Economic Bias
- Political Bias
- Infrastructure Bias


Associate each stereotypical comment with specific target states/regions

**4. LLM Classification Pipeline**

Zero-shot Classification:

Evaluate multiple LLMs' ability to classify biases without examples. Test different prompting strategies to optimise zero-shot performance.

Few-shot Classification:

Design effective examples for in-context learning and test varying numbers of examples to determine an optimal few-shot configuration.

**5. Performance Evaluation**

Create benchmark metrics to evaluate LLM performance:

Accuracy, Precision, Recall, F1-score

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

- [Data](https://github.com/debby123p/Detecting-Indian-Regional-Biases-in-Online-Social-Interactions-using-Large-Language-Models/blob/main/Miscellaneous/Regional_bias_data.csv)

The breakdown of the data among social media platforms is the following:

- From Reddit(9000 comments)

- From YouTube(1000 comments)

**Level-1**

![image](https://github.com/user-attachments/assets/ba4fe26b-d17f-41d8-822b-9da3f3879e62)

This chart shows the distribution between non-biased comments (7,580) and comments containing regional bias (2,420). The visualisation demonstrates that about a quarter of the analysed comments show some form of regional bias.

**Level-2**

![image](https://github.com/user-attachments/assets/fa4d4ce6-8e41-4220-b755-0cdca0884499)

Among biased comments, moderate severity (2) is most prevalent, followed by mild (1), with severe biases (3) being the least common but still significant. This chart visualizes the severity distribution of the comments. The data shows that moderate severity comments (1,328) are most common, followed by mild (872), with severe comments (220) being less frequent but still notable.

**Level-3**

![image](https://github.com/user-attachments/assets/08f9beb9-eb1c-473f-b6d6-0147abf0b39e)


When bias is present, it most commonly manifests as a single bias type, with declining frequency as the number of simultaneous bias types increases. This visualisation shows how many different types of biases appear in comments. Most biased comments contain just one type of bias (1,883), followed by two types (368), three types (156), and four types (60)

**Level-4**

![image](https://github.com/user-attachments/assets/d61fa854-ebcc-4e19-88fc-35d89d9da870)

The visualisation shows that Chhattisgarh (183) has the highest number of comments, Bihar (169), and Karnataka (166). The chart displays the top 15 states/regions in descending order, making it easy to understand the distribution of comments across different Indian states and regions.

![image](https://github.com/user-attachments/assets/eed151ef-e404-452e-9f0e-aeea9bda34c6)

The distribution shows North and South India leading with the highest engagement (716 and 677 comments, respectively), while Northeast India has the lowest participation (117 comments).

## Prompting and Classification Strategies

In the evaluation of the model's understanding of regional biases, we utilised the prompting techniques. This involved the careful curation of instructions for specific tasks into prompts that include context, definitions, and examples. We explored different prompting paradigms:

**Zero-Shot Classification:**

A classification technique that refers to the ability of large language models to perform classification solely based on the instructions and class descriptions provided in the prompt, without any seen examples of the specific task, unlike the fine-tuning technique.


- [LLaMa_3.2_3B](https://github.com/debby123p/Regional-Bias-Detection-for-Indian-States-using-Large-Language-Models/blob/main/zero_shot_models/llama_3.2_3b.py)
- [Deepseek_R1_Distill_Qwen_1.5B](https://github.com/debby123p/Summarizing-personalised-social-biases-towards-Indian-states-using-different-LLMs/blob/main/zero_shot_models/deepseek_r1_distill_qwen_7b.py)
- [Gemma_3_4B_it](https://github.com/debby123p/Regional-Bias-Detection-for-Indian-States-using-Large-Language-Models/blob/main/zero_shot_models/gemma_3_4b_it.py)
- [Mistral_7B_instruct_v0.3](https://github.com/debby123p/Regional-Bias-Detection-for-Indian-States-using-Large-Language-Models/blob/main/zero_shot_models/mistral_7b_instruct_v0.3.py)
- [Qwen_2.5_7B_instruct](https://github.com/debby123p/Regional-Bias-Detection-for-Indian-States-using-Large-Language-Models/blob/main/zero_shot_models/qwen_2.5_7b_instruct.py)
- [Deepseek_R1_Distill_Qwen_7B](https://github.com/debby123p/Regional-Bias-Detection-for-Indian-States-using-Large-Language-Models/blob/main/zero_shot_models/deepseek_r1_distill_qwen_7b.py)

**Few-Shot Classification:**

The classification techniques involve the use of a smaller number of examples within the prompt itself. These examples illustrate the desired input-output behaviour for the task. These examples help the model better adapt its pre-trained knowledge to specific nuances in the classification. 

**Few-Shot (Support-100)**

The examples are randomly selected from the annotated dataset, that is, **50 comments tagged regional biases and 50 comments tagged as non-regional biases**, as input to the prompt as support examples, helping the model to understand the nuances better.

- [Deepseek_R1_Distill_Qwen_1.5B](https://github.com/debby123p/Summarizing-personalised-social-biases-towards-Indian-states-using-different-LLMs/blob/main/few_shot_100_examples/deepseek_r1_distill_qwen_1.5b.py)
- [Mistral_7B_instruct_v0.3](https://github.com/debby123p/Summarizing-personalised-social-biases-towards-Indian-states-using-different-LLMs/blob/main/few_shot_100_examples/mistral_7b_instruct_v0.3.py)
- [Qwen_2.5_7B_instruct](https://github.com/debby123p/Regional-Bias-Detection-for-Indian-States-using-Large-Language-Models/blob/main/few_shot_100_examples/qwen_2.5_7b_instruct.py)
- [Deepseek_R1_Distill_Qwen_7B](https://github.com/debby123p/Regional-Bias-Detection-for-Indian-States-using-Large-Language-Models/blob/main/few_shot_150_examples/deepseek_r1_distill_7b.py)

**Few-Shot (Support-150)**

The examples are randomly selected from the annotated dataset, that is, **75 comments tagged regional biases and 75 comments tagged as non-regional biases**, as input to the prompt as support examples, helping the model to understand the nuances better.

- [Deepseek_R1_Distill_Qwen_1.5B](https://github.com/debby123p/Regional-Bias-Detection-for-Indian-States-using-Large-Language-Models/blob/main/few_shot_150_examples/deepseek_r1_distill_qwen_1.5b.py)
- [Mistral_7B_instruct_v0.3](https://github.com/debby123p/Regional-Bias-Detection-for-Indian-States-using-Large-Language-Models/blob/main/few_shot_150_examples/Mistral_7b_instruct_v0.3.py)
- [Qwen_2.5_7B_instruct](https://github.com/debby123p/Regional-Bias-Detection-for-Indian-States-using-Large-Language-Models/blob/main/few_shot_150_examples/qwen_2.5_7b_instruct.py)
- [Deepseek_R1_Distill_Qwen_7B](https://github.com/debby123p/Regional-Bias-Detection-for-Indian-States-using-Large-Language-Models/blob/main/few_shot_150_examples/deepseek_r1_distill_7b.py)
  
**Few-Shot (Support-200)**

The examples are randomly selected from the annotated dataset, that is, **100 comments tagged regional biases and 100 comments tagged as non-regional biases**, as input to the prompt as support examples, helping the model to understand the nuances better.

- [Deepseek_R1_Distill_Qwen_1.5B](https://github.com/debby123p/Regional-Bias-Detection-for-Indian-States-using-Large-Language-Models/blob/main/few_shot_200_examples/Deepseek_r1_distill_1.5b.py)
- [Mistral_7B_instruct_v0.3](https://github.com/debby123p/Regional-Bias-Detection-for-Indian-States-using-Large-Language-Models/blob/main/few_shot_200_examples/Mistral_7b_instruct_v0.3.py)
- [Qwen_2.5_7B_instruct](https://github.com/debby123p/Regional-Bias-Detection-for-Indian-States-using-Large-Language-Models/blob/main/few_shot_200_examples/Qwen_2.5_7b_instruct.py)
- [Deepseek_R1_Distill_Qwen_7B](https://github.com/debby123p/Regional-Bias-Detection-for-Indian-States-using-Large-Language-Models/blob/main/few_shot_200_examples/Deepseek_r1_distill_7b.py)

## Acknowledgement

We would like to formally recognise the contributions of Claude, which served as an AI-powered coding assistant, providing significant support during the development of this project



    



