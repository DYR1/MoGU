import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation.utils import GenerationConfig
from tqdm import tqdm
from transformers.generation.utils import GenerationConfig
import torch
from peft import (
    LoraConfig,
    get_peft_model,
    PeftModel
)
from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers.optimization import AdamW
import json


vicuna_template='<s> A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user\'s questions. USER: {} ASSISTANT:'
tokenizer = AutoTokenizer.from_pretrained("lmsys/vicuna-7b-v1.5", use_fast=False, trust_remote_code=True)
model = AutoModelForCausalLM .from_pretrained("lmsys/vicuna-7b-v1.5", device_map="auto",trust_remote_code=True)

max_length = 1024
train_on_inputs=False

def tokenize(prompt, add_eos_token=False):
    result = tokenizer(
        prompt,
        truncation=True,
        add_special_tokens=False,
        max_length=1024,
        padding=False,
        return_tensors=None,
    )
    if (
            result["input_ids"][-1] != tokenizer.eos_token_id
            and len(result["input_ids"]) < max_length
            and add_eos_token
    ):
        result["input_ids"].append(tokenizer.eos_token_id)
        result["attention_mask"].append(1)

    if add_eos_token and len(result["input_ids"]) >= max_length:
        result["input_ids"][max_length - 1] = tokenizer.eos_token_id
        result["attention_mask"][max_length - 1] = 1

    result["labels"] = result["input_ids"].copy()
    return result

def generate_prompt(instruction,input,label):
    if input:
        res = vicuna_template.format(instruction+input)
    else:
        res = vicuna_template.format(instruction)
    if label:
        res = f"{res}{label}"
    return res


def generate_and_tokenize_prompt(data_point):

    full_prompt=generate_prompt(
            data_point["instruction"],
            None,
            data_point["output"],
        )

    tokenized_full_prompt = tokenize(full_prompt)

    if not train_on_inputs:
        user_prompt = generate_prompt(
            data_point["instruction"], None, None
        )
        tokenized_user_prompt = tokenize(user_prompt, add_eos_token=False)
        user_prompt_len = len(tokenized_user_prompt["input_ids"])

        tokenized_full_prompt["labels"] = [
            -100
        ] * user_prompt_len + tokenized_full_prompt["labels"][
            user_prompt_len:
        ]  # could be sped up, probably`


    return tokenized_full_prompt

data = load_dataset("json", data_files="./data/safety_affirm.json")
train_data_affirm = data['train'].map(generate_and_tokenize_prompt)

data = load_dataset("json", data_files="./data/safety_reject.json")
train_data_reject = data['train'].map(generate_and_tokenize_prompt)

config = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    bias="none",
    target_modules=['o_proj'],
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, config)
for name, param in model.named_parameters():
    if param.requires_grad:
        print(name)

device = torch.device("cuda")
train_loader_affirm=DataLoader(train_data_affirm, shuffle=False, batch_size=1)
train_loader_reject=DataLoader(train_data_reject, shuffle=False, batch_size=1)
optimizer = AdamW(model.parameters(), lr=4e-5)
batches_affirm = tqdm(train_loader_affirm)
batches_reject = tqdm(train_loader_reject)

num_epochs=10
batch_size=16
cnt=0

loss_all=torch.tensor([0.0]).to(device)
optimizer.zero_grad()
for epoch in range(0,num_epochs):
    
    for batch_affirm, batch_reject in tqdm(zip(batches_affirm, batches_reject), total=len(batches_affirm)):
        
        input_ids,attention_masks,labels=torch.tensor([batch_affirm['input_ids']]).to(device),torch.tensor([batch_affirm['attention_mask']]).to(device),torch.tensor([batch_affirm['labels']]).to(device)
        loss_affirm= model(input_ids=input_ids, attention_mask=attention_masks, labels=labels).loss
        input_ids,attention_masks,labels=torch.tensor([batch_reject['input_ids']]).to(device),torch.tensor([batch_reject['attention_mask']]).to(device),torch.tensor([batch_reject['labels']]).to(device)
        loss_reject= model(input_ids=input_ids, attention_mask=attention_masks, labels=labels).loss
        loss_all+=(loss_reject/loss_affirm)

        if cnt!=0 and cnt%batch_size==0:
            loss_mean=loss_all/batch_size
            print(loss_mean)
            loss_mean.backward()
            optimizer.step()
            optimizer.zero_grad()
            loss_all=torch.tensor([0.0]).to(device)
        
        if cnt!=0 and cnt%80==0:
            output_dir='./resp_unwill/'+str(cnt)+'_lora'
            model.save_pretrained(output_dir)
        cnt+=1


