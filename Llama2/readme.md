We provide the MoGU training code for Llama2. You can run the following code.
```python
python sft_glad.py
python sft_unwill.py
python sft_mogu.py
```
The parameters we trained can be found at [https://huggingface.co/yrdu/mogu_param]. You can put our trained parameters into the corresponding folder(resp_glad/resp_unwill/router_layer) and run the following inference code.
```python
python inf_mogu.py
```
