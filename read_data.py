from transformers import AutoTokenizer
import os
import torch
import argparse


def load_data(args):
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    fn = os.path.join('data', os.path.basename(args.model_name), 
        f'{args.quant_type}-{args.split}_error_seqs-10.pt')

    if os.path.exists(fn):
        print('Load data from:', fn)
        tokenized_data = torch.load(fn)
        print('Shape:', tokenized_data.shape)
        texts = tokenizer.batch_decode(tokenized_data) 
        return tokenized_data, texts
    else:
        print(fn, 'not found!')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True,
        choices=['meta-llama/Meta-Llama-3-8B', 'meta-llama/Meta-Llama-3-70B', 
                'Qwen/Qwen2.5-7B', 'mistralai/Mistral-Nemo-Base-2407'])
    parser.add_argument("--split", type=str, choices=['large', 'nonlarge'], required=True)
    parser.add_argument("--quant_type", type=str, default='awq3', choices=['awq3', 'gptq3', 'nf3'])
    args = parser.parse_args()

    tokenized_data, texts = load_data(args)
