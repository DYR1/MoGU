# MoGU
Our research proposes a novel MoGU framework that improves LLMs' safety while preserving their usability.

## Paper (Under Review)
MoGU: A Framework for Enhancing Safety of Open-Sourced LLMs While Preserving Their Usability
[https://arxiv.org/abs/2405.14488]

Open-sourced Parameters: https://huggingface.co/yrdu

## Abstract 
Large Language Models (LLMs) are increasingly deployed in various applications. As their usage grows, concerns regarding their safety are rising, especially in maintaining harmless responses when faced with malicious instructions. 
Many defense strategies have been developed to enhance the safety of LLMs. However, our research finds that existing defense strategies lead LLMs to predominantly adopt a rejection-oriented stance, thereby diminishing the usability of their responses to benign instructions. To solve this problem, we introduce the MoGU framework, designed to enhance LLMs' safety while preserving their usability. Our MoGU framework transforms the base LLM into two variants: the usable LLM and the safe LLM, and further employs dynamic routing to balance their contribution. When encountering malicious instructions, the router will assign a higher weight to the safe LLM to ensure that responses are harmless. Conversely, for benign instructions, the router prioritizes the usable LLM, facilitating usable and helpful responses. On various open-sourced LLMs, we compare multiple defense strategies to verify the superiority of our MoGU framework. Besides, our analysis provides key insights into the effectiveness of MoGU and verifies that our designed routing mechanism can effectively balance the contribution of each variant by assigning weights. Our work released the safer Llama2, Vicuna, Falcon, Dolphin, and Baichuan2.

## Open-sourced LLMs and Code
We currently open-sourced three safer LLMs,  including Llama2, Vicuna, and Falcon. You can find parameters and inference code at https://huggingface.co/yrdu. The current inference code is still in a simple version and we will further improve it. In the future, we plan to open-source training data, training code, and other safer LLMs.
