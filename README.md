## Why Do Some Inputs Break Low-Bit LLM Quantization? (EMNLP 2025)

![awq](https://img.shields.io/badge/AWQ-green.svg?style=plastic)
![gptq](https://img.shields.io/badge/GPTQ-green.svg?style=plastic)
![NF](https://img.shields.io/badge/NF-green.svg?style=plastic)

> Ting-Yun Chang, Muru Zhang, Jesse Thomason, and Robin Jia<br>

> **Paper**: https://arxiv.org/abs/2506.12044

## Data
- We release $D_\text{large}$ and $D_\text{ctrl}$ under [data/](https://github.com/terarachang/QError/tree/master/data)
- The data are tokenized, where each split has shape [1000, 512], and each row corresponds to a [FineWeb](https://huggingface.co/datasets/HuggingFaceFW/fineweb) example with sequence length = 512.
- To convert them back into texts, run: 
`python read_data.py --split large --quant_type awq3 --model_name Qwen/Qwen2.5-7B`
