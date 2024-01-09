models = {
    0: {'name': 'bert/distilbert-base-uncased-finetuned-sst-2-english', 'gptq': False},
    1: {'name': 'mistralai/Mistral-7B-v0.1', 'gptq': False},
    2: {'name': 'mistralai/Mistral-7B-Instruct-v0.1', 'gptq': False},
    3: {'name': 'teknium/OpenHermes-2.5-Mistral-7B', 'gptq': False},
    4: {'name': 'TheBloke/OpenHermes-2.5-Mistral-7B-GPTQ', 'gptq': True},
    # 5: {'name': 'TheBloke/neural-chat-7B-v3-2-GPTQ', 'gptq': True},
}

datasets = {
    0: "mlabonne/guanaco-llama2-1k"
}

finetunes = {
    0: {'name': 'mistralai/Mistral-7B-Instruct-v0.1/finetunes/mlabonne/guanaco-llama2-1k', 'gptq': False}
}
