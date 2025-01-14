from transformers.generation.utils import GenerationConfig
import torch
from safetensors import safe_open
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers.optimization import AdamW
from datasets import load_dataset
from modeling_llama_mogu import LlamaForCausalLM
from transformers import AutoTokenizer
import json


tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf", use_fast=False, trust_remote_code=True)
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
        res = llama_template.format(instruction+input)
    else:
        res = llama_template.format(instruction)
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

llama_template="<s>[INST] <<SYS>>\nYou are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.\n\nIf a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.\n<</SYS>>\n\n{}[/INST]"
model_lora = LlamaForCausalLM.from_pretrained("meta-llama/Llama-2-7b-chat-hf", device_map="auto", trust_remote_code=True)

lora_0_A={}
lora_0_B={}
model_path = './resp_unwill/960_lora/adapter_model.safetensors'
tensors = {}
with safe_open(model_path, framework="pt", device='cpu') as f:
    for k in f.keys():
        tensors[k] = f.get_tensor(k)
for k,v in tensors.items():
    ks=k.split('.')
    if ks[7]=='lora_A':
        lora_0_A[ks[4]]=v
    if ks[7]=='lora_B':
        lora_0_B[ks[4]]=v

lora_1_A={}
lora_1_B={}
model_path = './resp_glad/800_lora/adapter_model.safetensors'
tensors = {}
with safe_open(model_path, framework="pt", device='cpu') as f:
    for k in f.keys():
        tensors[k] = f.get_tensor(k)
for k,v in tensors.items():
    ks=k.split('.')
    if ks[7]=='lora_A':
        lora_1_A[ks[4]]=v
    if ks[7]=='lora_B':
        lora_1_B[ks[4]]=v

for name, param in model_lora.named_parameters():
    print(name)
    ns=name.split('.')
    if len(ns)>=5 and ns[3]=='self_attn' and ns[4]=='lora_0' and ns[5]=='linear1':
        param.data=lora_0_A[ns[2]].clone().detach().cuda()
    if len(ns)>=5 and ns[3]=='self_attn' and ns[4]=='lora_0' and ns[5]=='linear2':
        param.data=lora_0_B[ns[2]].clone().detach().cuda()
    if len(ns)>=5 and ns[3]=='self_attn' and ns[4]=='lora_1' and ns[5]=='linear1':
        param.data=lora_1_A[ns[2]].clone().detach().cuda()
    if len(ns)>=5 and ns[3]=='self_attn' and ns[4]=='lora_1' and ns[5]=='linear2':
        param.data=lora_1_B[ns[2]].clone().detach().cuda()


for name, param in model_lora.named_parameters():
    ns=name.split('.')
    if ns[1] not in ['routers']:
        param.requires_grad=False

data = load_dataset("json", data_files="./data/data_normal.json")
train_data_normal = data['train'].map(generate_and_tokenize_prompt)
alpha=2.0


f=open('./data/data_label.json','r',encoding='utf-8')
d_label=json.load(f)

device = torch.device("cuda")
train_loader_normal=DataLoader(train_data_normal, shuffle=False, batch_size=1)
optimizer = AdamW(model_lora.parameters(), lr=5e-4)

for name, param in model_lora.named_parameters():
    if param.requires_grad==True:
        print(name)

batches_normal = tqdm(train_loader_normal)
num_epochs=10
gradient_accumulation_steps=16
cnt=0
loss_all=torch.tensor([0.0],dtype=torch.bfloat16).to(device)

optimizer.zero_grad()
for epoch in range(0,num_epochs):
    for batch_normal,label in tqdm(zip(batches_normal, d_label), total=len(batches_normal)):
        
        input_ids,attention_masks,labels=torch.tensor([batch_normal['input_ids']]).to(device),torch.tensor([batch_normal['attention_mask']]).to(device),torch.tensor([batch_normal['labels']]).to(device)
        loss_normal= model_lora(input_ids=input_ids, attention_mask=attention_masks, labels=labels).loss
        if label == 0:
            target_ones = torch.ones(model_lora.model.alphas.size()).cuda()
            target_zeros = torch.zeros(model_lora.model.alphas.size()).cuda()
            loss_alpha= (model_lora.model.alphas - target_zeros).pow(2).mean()
            loss_beta= (model_lora.model.betas - target_ones).pow(2).mean()
            loss_all=loss_all+loss_normal+alpha*(loss_alpha+loss_beta)
        if label == 1:
            target_ones = torch.ones(model_lora.model.alphas.size()).cuda()
            target_zeros = torch.zeros(model_lora.model.alphas.size()).cuda()
            loss_alpha= (model_lora.model.alphas - target_ones).pow(2).mean()
            loss_beta= (model_lora.model.betas - target_zeros).pow(2).mean()
            loss_all=loss_all+loss_normal+alpha*(loss_alpha+loss_beta)

        if  cnt!=0 and cnt%gradient_accumulation_steps==0:
            
            loss_mean=loss_all/gradient_accumulation_steps
            print(loss_mean)

            loss_mean.backward()
            optimizer.step()
            optimizer.zero_grad()
            loss_all=torch.tensor([0.0],dtype=torch.bfloat16).to(device)
        
        if cnt!=0 and cnt%80==0:
            if cnt==1920 or cnt==3920 or cnt==5920:
                to_save = {k: v for k, v in model_lora.state_dict().items() if 'routers' in k}
                torch.save(to_save, './router_layer/'+str(cnt)+'_router.pth')
        cnt+=1
