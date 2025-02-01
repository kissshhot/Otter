<!-- # Otter-->

[![Magpie](figs/magpie_logo.png)](https://magpie-align.github.io/)

[![arXiv](https://img.shields.io/badge/arXiv-paper-b31b1b.svg)](https://arxiv.org/abs/2406.08464) [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) [![Data License](https://img.shields.io/badge/Data%20License-CC%20By%20NC%204.0-red.svg)](https://huggingface.co/Magpie-Align) [![Spaces](https://img.shields.io/badge/🤗-Open%20in%20Spaces-blue)](https://huggingface.co/spaces/davanstrien/magpie)

This is the official repository for ICLR 2025 paper "[Otter: Browsing then Ideating Helps in Document-Inspired High-Quality Instruction Data Synthesis](https://arxiv.org/abs/2406.08464)". Magpie generates high-quality alignment data by prompting aligned LLMs with their pre-query templates. Unlike many existing synthetic data generation methods, Magpie doesn't rely on prompt engineering or seed questions for generating synthetic data. Instead, it uses the prompt template of an aligned LLM to generate both the user query and an LLM response.

- 🤗 [**Huggingface (Models and Datasets)**](https://huggingface.co/datasets/kissshhot)
- 🧭 [**Dataset Navigation**](navigation.md)
- 🕸️ [**Website**](https://magpie-align.github.io/)
- 📄 [**Technical Report**](https://arxiv.org/abs/2406.08464)
- 🤗 [**Magpie Demo**](https://huggingface.co/spaces/davanstrien/magpie) (Thanks a lot for the implementation from @davanstrien!)
- 🐦 [**Chat with Magpie**](https://huggingface.co/spaces/flydust/Chat-with-Magpie)

## 🐦 News

## Magpie Supports


|Model Family | Magpie | Magpie Scripts | Datasets | Size |
|-------------|:------:|:-------|:-------|:-------|
| [Llama 3.3](https://huggingface.co/collections/meta-llama/llama-33-67531d5c405ec5d08a852000)     | ✅ | [70B](scripts/magpie-llama3.3-70b.sh) | [70B](https://huggingface.co/datasets/Magpie-Align/Magpie-Llama-3.3-Pro-1M-v0.1) | 1M |
| [Llama 3.1](https://huggingface.co/collections/meta-llama/llama-31-669fc079a0c406a149a5738f)     | ✅ * | [8B](scripts/magpie-llama3.1-8b.sh),[70B](scripts/magpie-llama3.1-70b.sh) | [70B](https://huggingface.co/collections/Magpie-Align/magpie-llama31-datasets-66a45ed727be07f53c8ff294),[405B(Argilla)](https://huggingface.co/datasets/argilla/magpie-ultra-v0.1) | 1M |
| [Llama 3](https://huggingface.co/collections/meta-llama/meta-llama-3-66214712577ca38149ebb2b6)     | ✅ | [8B](scripts/magpie-llama3-8b.sh),[70B](scripts/magpie-llama3-70b.sh) | [8B](https://huggingface.co/collections/Magpie-Align/magpie-air-6666b11a32021655a27f86c0),[70B](https://huggingface.co/collections/Magpie-Align/magpie-pro-6666b0e713e5f5c09554876f) | 3M + 1M |
| [Qwen2.5](https://huggingface.co/collections/Qwen/qwen25-66e81a666513e518adb90d9e)     | ✅ | [3B](scripts/magpie-qwen2.5-3b.sh),[7B](scripts/magpie-qwen2.5-7b.sh),[14B](scripts/magpie-qwen2.5-14b.sh),[32B](scripts/magpie-qwen2.5-32b.sh),[72B](scripts/magpie-qwen2.5-72b.sh) | [72B](https://huggingface.co/datasets/Magpie-Align/Magpie-Qwen2.5-Pro-1M-v0.1) | 1M | 

- ✅: It works great! (**\*** Apply a logits processor to surpress markdown; **\*\*** Apply a [filter](exp/str_utils.py) before generating responses.)
- ⭕️: It works! We can get something interesting, but we may need to design an additional logit processor and/or a filter.
- ❌: Not work.
- ❓: Untested.

The navigation of all available Magpie datasets can be found [here](navigation.md).

We hope Magpie can contribute to the democratization of AI with enhanced transparency of model alignment processes!

## Abstract
<details><summary>Click Here</summary>
High-quality instruction data is critical for aligning large language models (LLMs).
</details><be>

## Overview

![Overview](figs/overview.png)

## Installation

**Build environment**
```
git clone https://github.com/magpie-align/magpie.git
cd magpie
conda create -n magpie python=3.10 -y
conda activate magpie
pip install -r requirements.txt
```

**Get access to Llama-3 models from 🤗 Huggingface**

You can apply for Llama-3 model access [here](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct). To login in the terminal, enter:
```
huggingface-cli login
```
then enter your Huggingface private key beginning with "hf_".

## Toy Example


## Batched SFT Data Generation
We use Llama-3-8B-Instruct as an example to demonstrate the batched SFT data generation process. To run batched generation, you can simply run:
```
cd scripts
bash magpie.sh
```
The script will generate both instructions and responses in the data folder. It has been tested on an RTX 4090 24G GPU. If you are using GPUs with less memory, consider implementing [quantization](https://docs.vllm.ai/en/latest/quantization/fp8.html).

We also provide scripts for other models in the [`scripts`](scripts) folder. You can use [this](#magpie-supports) navigation to find specific Magpie scripts. Note that for model sizes greater than 8B, you may need 4*A100 GPUs to run the scripts.

### Batched Multi-turn Data Generation \[Optional\]
After generating instruction-response pairs, you can extend them to multi-turn conversations. To do so, simply run the following command:
```
bash magpie-multi-turn.sh ***_ins_res.json
```
where `***_ins_res.json` is the single-turn instruction-response pairs generated in the previous step.

## Dataset Filtering
### 1. Tagging
To tag the generated instruction-response pairs, you can run:
```
cd scripts
bash unitag.sh ***_ins_res.json all
```
This script will automatically generate quality, difficulty, task category, safety, reward, and language for the generated dataset. You can also generate one tag at a time. For example, if you just want to generate the safety label using device 0, you can run:
```
cd scripts
bash unitag.sh ***_ins_res.json safety 0
```
### 2. Data Concatenation and Converting
You may generate datasets with different generation configurations. We provide a Jupyter notebook [here](data_sft/data_concatenation.ipynb) for concatenating all datasets and converting them to ShareGPT format, which is fully supported by [Axolotl](https://github.com/OpenAccess-AI-Collective/axolotl) for fine-tuning.

### 3. Removing Repetition
Once you have a full dataset converted to ShareGPT format, you can calculate the minimum neighbor distance of each instruction and remove repetitions. To do so, run:
```
cd exp
python gen_dis.py --input_file ***_sharegpt.jsonl
```
where `***_sharegpt.jsonl` is the dataset path obtained in the previous step. The Python script will take care of building the FAISS index and calculating the minimum distance. 

### 4. Design and Apply Your Filter
We provide a Jupyter notebook [here](data_sft/data_filter.ipynb) for simple filtering. You can adjust the filtering parameters to design and apply your own filter based on your needs.

## Preference Data Generation

To generate preference data, first prepare filtered instructions following the steps outlined above. For the expected format, please refer to our example [here](data_po/example_instructions.jsonl).

Next, please use our provided scripts [here](scripts/magpie_example_po.sh) to generate multiple responses and compute their corresponding rewards. Finally, your can process the data and upload it to Huggingface using [this Jupyter notebook](data_po/process_po.ipynb).

## Fine-tuning
Please take a look at the [recipes](recipes/) directory for instructions and our Magpie model recipes.

## Citation

If you find the model, data, or code useful, please cite our paper 🤩:
```
@article{xu2024magpie,
  title={Magpie: Alignment Data Synthesis from Scratch by Prompting Aligned LLMs with Nothing},
  author={Zhangchen Xu and Fengqing Jiang and Luyao Niu and Yuntian Deng and Radha Poovendran and Yejin Choi and Bill Yuchen Lin},
  journal={ArXiv},
  year={2024},
  volume={abs/2406.08464},
  url={https://api.semanticscholar.org/CorpusID:270391432}
}
```

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=magpie-align/magpie&type=Date)](https://star-history.com/#magpie-align/magpie&Date)
