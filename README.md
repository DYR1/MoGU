# News
- [Coming Soon] We have developed a lighter MoGU framework (MoGU V2), which requires less inference cost and achieves better security.
- [Coming Soon] We propose a novel method for secure fine-tuning, which focuses on preserving LLMs' security during the fine-tuning phase (The [Arxiv Version](https://arxiv.org/abs/2410.04524) is just a preliminary version. We have designed a new and more effective method.).
- [2025.1.15] We released MoGU's training data and code for Llama2, Falcon, and Vicuna.
- [2024.12.10] Our another work (Arxiv Version: [Analyzing the Inherent Response Tendency of LLMs: Real-World Instructions-Driven Jailbreak](https://arxiv.org/abs/2312.04127)) was accepted by AAAI 2025. This work investigates the security threat arising from “Yes-No” implicit bias in LLMs.
- [2024.9.26] Our MoGU work ([MoGU: A Framework for Enhancing Safety of Open-Sourced LLMs While Preserving Their Usability](https://openreview.net/pdf?id=SrFbgIjb53)) was accepted by NeurIPS 2024 and we released MoGU's inference [code](https://huggingface.co/yrdu).
- [2024.5.23] Our research proposed a novel MoGU framework that improves LLMs' safety while preserving their usability.

## MoGU's Abstract 
Large Language Models (LLMs) are increasingly deployed in various applications. As their usage grows, concerns regarding their safety are rising, especially in maintaining harmless responses when faced with malicious instructions. 
Many defense strategies have been developed to enhance the safety of LLMs. However, our research finds that existing defense strategies lead LLMs to predominantly adopt a rejection-oriented stance, thereby diminishing the usability of their responses to benign instructions. To solve this problem, we introduce the MoGU framework, designed to enhance LLMs' safety while preserving their usability. Our MoGU framework transforms the base LLM into two variants: the usable LLM and the safe LLM, and further employs dynamic routing to balance their contribution. When encountering malicious instructions, the router will assign a higher weight to the safe LLM to ensure that responses are harmless. Conversely, for benign instructions, the router prioritizes the usable LLM, facilitating usable and helpful responses. On various open-sourced LLMs, we compare multiple defense strategies to verify the superiority of our MoGU framework. Besides, our analysis provides key insights into the effectiveness of MoGU and verifies that our designed routing mechanism can effectively balance the contribution of each variant by assigning weights. 

## How to train and infer?
Install Enviroment
```python
pip install -r requiement.txt
```
Take Llama2 for example.
We provide the MoGU training code for Llama2. You can run the following code.
```python
cd ./Llama2
python sft_glad.py
python sft_unwill.py
python sft_mogu.py
```
The parameters we trained can be found at [https://huggingface.co/yrdu/mogu_param]. You can put our trained parameters into the corresponding folder(resp_glad/resp_unwill/router_layer) and run the following inference code.
```python
python inf_mogu.py
```

## About hyperparameter parameters
All hyperparameter parameters have been fixed in our training code, and the corresponding results have been reported in our paper.
If you want to adjust the hyperparameters, according to our experience:
- The choice of **checkpoint** for glad responder and unhappy responder will have a greater impact on the experimental results.
- The choice of hyperparameter parameter **alpha** in sft_mogu.py will have a greater impact on the experimental results.


## Statement
The author is currently preparing for the ACL submission, so he has not spent too much effort on the MoGU's training code. The current version of the training code is relatively simple, so please bear with me. If you have any questions about the code, please email me (yrdu@ir.hit.edu.cn).
