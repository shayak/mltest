import os
import shutil

import torch
import pynvml
from huggingface_hub import (
    hf_hub_download
)
from transformers import (
    pipeline,
    BitsAndBytesConfig,
    # MistralForCausalLM,
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer
)
from pathlib import Path
from huggingface import defs

# BASE_DIR = str(Path.home()) + '/dev/llms/models'
BASE_DIR = str(Path.home()) + '/data/llms/models/huggingface'
BASE_CACHE = str(Path.home()) + '/.cache/huggingface/hub'
MODEL_NUM = 1

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# DEVICE = 'cpu'
print(DEVICE)


def _init():
    path = BASE_DIR
    dirs = []
    _ = [dirs.extend([f'{d}/{dir}' for dir in os.listdir(f'{path}/{d}')]) for d in os.listdir(path)]

    keep_set = set([mod['name'] for mod in defs.models.values()])
    existing_set = set(dirs)
    remove_set = existing_set.difference(keep_set)
    add_set = keep_set.difference(existing_set)
    print(f'existing: {existing_set}\nkeep: {keep_set}\nremove: {remove_set}\nadd: {add_set}')

    # delete remove_set
    for mname in remove_set:
        rem_path = f'{path}/{mname}'
        print(f'Removing model: {rem_path}')
        shutil.rmtree(rem_path)

    # download_and_save add_set
    for mname in add_set:
        print(f'Downloading model: {mname}')
        download_model(mname)


def download_model(model_name):
    print('downloading...')
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map='cpu')
    # model.to(DEVICE)
    print('downloaded.')
    save_dir = BASE_DIR + f'/{model_name}'
    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)

    # delete the cached download
    rmdir = BASE_CACHE + f'/models--{model_name.replace("/", "--")}'
    print(rmdir)
    shutil.rmtree(rmdir)

    return model, tokenizer


def load_local_model(model_name, gptq=False):
    model_path = BASE_DIR + f'/{model_name}'
    print('loading...')

    # model_config = AutoConfig.from_pretrained(model_path)

    # model = MistralForCausalLM.from_pretrained(BASE_DIR + f'/{model_name}', device_map='auto')
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    # gptq models don't use bnb configs
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        # quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    ) if gptq else AutoModelForCausalLM.from_pretrained(
        model_path,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    # model.to(DEVICE)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    tokenizer.padding_side = 'right'
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.add_eos_token = True

    print('loaded.')

    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    info = pynvml.nvmlDeviceGetMemoryInfo(handle)
    print(f"GPU memory occupied: {info.used // 1024 ** 2} MB.")

    return model, tokenizer


def do_inference(model, tokenizer, prompt):
    # print(tokenizer.batch_decode(outputs))

    inputs = tokenizer.encode_plus(
        prompt,
        return_tensors="pt"
    )['input_ids'].to(DEVICE)

    outputs = model.generate(
        inputs,
        do_sample=True,
        max_length=5000,
        temperature=0.8,
        top_k=50,
        top_p=0.9,
        num_return_sequences=1
    )

    response = tokenizer.decode(outputs[0])
    return response


def chat_with_model(model, tokenizer):
    while True:
        prompt = input('\nEnter prompt:\n')
        if not prompt.strip():
            break

        response = do_inference(model, tokenizer, prompt)

        print(f'Response:\n{response}')


def run():
    mem_alloc_gpu = torch.cuda.memory_allocated('cuda')

    print(f'Memory allocated, GPU: {mem_alloc_gpu / (1024 ** 3)}')

    _init()

    model_config = defs.models[MODEL_NUM]

    model_name, gptq = model_config['name'], model_config['gptq']
    print(f'Model name: {model_name}, gptq: {gptq}')

    # model, tokenizer = download_and_save_model(model_name)
    model, tokenizer = load_local_model(model_name, gptq=gptq)


    # mem_alloc_cpu = torch.cuda.memory_allocated('cpu')
    mem_alloc_gpu = torch.cuda.memory_allocated('cuda')

    print(f'Memory allocated, GPU: {mem_alloc_gpu / (1024 ** 3)} GB')

    # Calculate the size of parameters to be moved to the CPU
    cpu_params_size = 0
    trainable_params = 0
    layer = 0
    for name, param in model.named_parameters():
        numel = param.numel()
        # print(layer, f'{name}', numel)
        if param.requires_grad:
            trainable_params += numel
        cpu_params_size += numel
        layer += 1

    print(f'Total params: {model.num_parameters()}, Numel: {cpu_params_size}, Trainable: {trainable_params}')

    chat_with_model(model, tokenizer)

    print('Exiting')


if __name__ == "__main__":
    run()
