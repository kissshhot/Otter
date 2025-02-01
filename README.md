<!-- # Otter-->

[![Magpie](figs/magpie_logo.png)](https://magpie-align.github.io/)

[![arXiv](https://img.shields.io/badge/arXiv-paper-b31b1b.svg)](https://arxiv.org/abs/2406.08464) [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) [![Data License](https://img.shields.io/badge/Data%20License-CC%20By%20NC%204.0-red.svg)](https://huggingface.co/Magpie-Align) [![Spaces](https://img.shields.io/badge/🤗-Open%20in%20Spaces-blue)](https://huggingface.co/spaces/davanstrien/magpie)

This is the official repository for ICLR 2025 paper "[Alignment Data Synthesis from Scratch by Prompting Aligned LLMs with Nothing](https://arxiv.org/abs/2406.08464)". Magpie generates high-quality alignment data by prompting aligned LLMs with their pre-query templates. Unlike many existing synthetic data generation methods, Magpie doesn't rely on prompt engineering or seed questions for generating synthetic data. Instead, it uses the prompt template of an aligned LLM to generate both the user query and an LLM response.

- 🤗 [**Huggingface (Models and Datasets)**](https://huggingface.co/Magpie-Align)
- 🧭 [**Dataset Navigation**](navigation.md)
- 🕸️ [**Website**](https://magpie-align.github.io/)
- 📄 [**Technical Report**](https://arxiv.org/abs/2406.08464)
- 🤗 [**Magpie Demo**](https://huggingface.co/spaces/davanstrien/magpie) (Thanks a lot for the implementation from @davanstrien!)
- 🐦 [**Chat with Magpie**](https://huggingface.co/spaces/flydust/Chat-with-Magpie)

## 🐦 News
- [2025/01/22] Magpie paper is accepted by ICLR 2025! 
- [2025/01/09] Magpie Reasoning V2 dataset is out! [250K]([https://huggiK](https://huggingface.co/collections/Magpie-Align/magpie-reasoning-datasets-67790a13b91035bc42693885)) from Llama, Skywork-o1 and QwQ! This time, we focus on CoT 🤯
- [2025/01/01] Magpie Llama-3.3 dataset is out! [1M](https://huggingface.co/datasets/Magpie-Align/Magpie-Llama-3.3-Pro-1M-v0.1) from Llama-3.3-70B-Instruct! Happy New Year!
- [2024/10/20] Magpie Qwen2.5 dataset is out! [1M](https://huggingface.co/datasets/Magpie-Align/Magpie-Qwen2.5-Pro-1M-v0.1) from Qwen2.5 72B!
- [2024/09/17] Ship two new models with SOTA performance: 𝙼𝚊𝚐𝚙𝚒𝚎𝙻𝙼-𝙲𝚑𝚊𝚝 (4B & 8B)! See collection [here](https://huggingface.co/collections/Magpie-Align/magpielm-66e2221f31fa3bf05b10786a)!
- [2024/08/19] Three preference optimization datasets, [Magpie-Air-DPO-100K-v0.1](https://huggingface.co/datasets/Magpie-Align/Magpie-Air-DPO-100K-v0.1), [Magpie-Pro-DPO-100K-v0.1](https://huggingface.co/datasets/Magpie-Align/Magpie-Pro-DPO-100K-v0.1), and [Magpie-Llama-3.1-Pro-DPO-100K-v0.1](https://huggingface.co/datasets/Magpie-Align/Magpie-Llama-3.1-Pro-DPO-100K-v0.1) are out! 
- [2024/07/25] Magpie Llama-3.1 dataset is out! [1M](https://huggingface.co/datasets/Magpie-Align/Magpie-Llama-3.1-Pro-1M-v0.1) from Llama-3.1-70B-Instruct! More friendly license compared with Llama-3 😃!
- [2024/07/21] Magpie Gemma2 dataset is out! [534K](https://huggingface.co/collections/Magpie-Align/magpie-gemma2-datasets-669da6aff21b09fdcecbd6ea) from Gemma-2-27b-it!
- [2024/07/19] [Llama-3-8B-Magpie-Align-v0.3](https://huggingface.co/Magpie-Align/Llama-3-8B-Magpie-Align-v0.3) is out with enhanced Chinese question-answering ability, thanks to our new [Chinese instruction dataset](https://huggingface.co/datasets/Magpie-Align/Magpie-Qwen2-Pro-200K-Chinese)!
- [2024/07/14] [Llama-3-8B-Magpie-Align-v0.2](https://huggingface.co/Magpie-Align/Llama-3-8B-Magpie-Align-v0.2) is out with enhanced reasoning ability, thanks to our new [reasoning booster dataset](https://huggingface.co/datasets/Magpie-Align/Magpie-Reasoning-150K)!
- [2024/07/04] Magpie Qwen2 dataset is out! [1M](https://huggingface.co/datasets/Magpie-Align/Magpie-Qwen2-Pro-1M-v0.1) from Qwen2 72B and [3M](https://huggingface.co/datasets/Magpie-Align/Magpie-Qwen2-Air-3M-v0.1) from Qwen2 7B.
- [2024/07/03] 🏆 Our open aligned model, [Llama-3-8B-Magpie-Align-v0.1](https://huggingface.co/Magpie-Align/Llama-3-8B-Magpie-Align-v0.1) is out! It is 🏆 the **best <30B Model** in [AI2 WildBench Leaderboard](https://huggingface.co/spaces/allenai/WildBench)! Even better than the official [Meta-Llama-3-8B-Instruct](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct) model!
- [2024/06/24] Magpie Phi 3 dataset is out! [1M](https://huggingface.co/collections/Magpie-Align/magpie-phi3-667a7a45f1a406cd61685d64) from Phi 3 Medium.
- [2024/06/12] Magpie Llama-3 dataset is out! [1M](https://huggingface.co/collections/Magpie-Align/magpie-pro-6666b0e713e5f5c09554876f) from Llama-3 70B and [3M](https://huggingface.co/collections/Magpie-Align/magpie-air-6666b11a32021655a27f86c0) from Llama-3 8B.
- [2024/06/12] [Magpie technical report]((https://arxiv.org/abs/2406.08464)) is out! Let's make high-quality alignment data open for all!

## Magpie Supports

Currently, Magpie has been tested on the **Llama-3**, **Qwen2**, **Phi 3** and **Gemma-2** series. Please [submit an issue](https://github.com/magpie-align/magpie/issues/new) for more model support.

|Model Family | Magpie | Magpie Scripts | Datasets | Size |
|-------------|:------:|:-------|:-------|:-------|
| [Llama 3.3](https://huggingface.co/collections/meta-llama/llama-33-67531d5c405ec5d08a852000)     | ✅ | [70B](scripts/magpie-llama3.3-70b.sh) | [70B](https://huggingface.co/datasets/Magpie-Align/Magpie-Llama-3.3-Pro-1M-v0.1) | 1M |
| [Llama 3.1](https://huggingface.co/collections/meta-llama/llama-31-669fc079a0c406a149a5738f)     | ✅ * | [8B](scripts/magpie-llama3.1-8b.sh),[70B](scripts/magpie-llama3.1-70b.sh) | [70B](https://huggingface.co/collections/Magpie-Align/magpie-llama31-datasets-66a45ed727be07f53c8ff294),[405B(Argilla)](https://huggingface.co/datasets/argilla/magpie-ultra-v0.1) | 1M |
| [Llama 3](https://huggingface.co/collections/meta-llama/meta-llama-3-66214712577ca38149ebb2b6)     | ✅ | [8B](scripts/magpie-llama3-8b.sh),[70B](scripts/magpie-llama3-70b.sh) | [8B](https://huggingface.co/collections/Magpie-Align/magpie-air-6666b11a32021655a27f86c0),[70B](https://huggingface.co/collections/Magpie-Align/magpie-pro-6666b0e713e5f5c09554876f) | 3M + 1M |
| [Qwen2.5](https://huggingface.co/collections/Qwen/qwen25-66e81a666513e518adb90d9e)     | ✅ | [3B](scripts/magpie-qwen2.5-3b.sh),[7B](scripts/magpie-qwen2.5-7b.sh),[14B](scripts/magpie-qwen2.5-14b.sh),[32B](scripts/magpie-qwen2.5-32b.sh),[72B](scripts/magpie-qwen2.5-72b.sh) | [72B](https://huggingface.co/datasets/Magpie-Align/Magpie-Qwen2.5-Pro-1M-v0.1) | 1M | 
| [Qwen2](https://huggingface.co/collections/Qwen/qwen2-6659360b33528ced941e557f)     | ✅ | [7B](scripts/magpie-qwen2-7b.sh),[72B](scripts/magpie-qwen2-72b.sh),[Math 7B](scripts/magpie-qwen2-math-7b.sh) | [7B](https://huggingface.co/datasets/Magpie-Align/Magpie-Qwen2-Air-3M-v0.1),[72B](https://huggingface.co/datasets/Magpie-Align/Magpie-Qwen2-Pro-1M-v0.1) | 3M + 1M |
| [Phi 3](https://huggingface.co/collections/microsoft/phi-3-6626e15e9585a200d2d761e3)     | ✅ | [mini](scripts/magpie-phi3mini.sh),[small](scripts/magpie-phi3small.sh),[medium](scripts/magpie-phi3medium.sh) | [medium](https://huggingface.co/collections/Magpie-Align/magpie-phi3-667a7a45f1a406cd61685d64) | 1M |
| [Gemma-2](https://huggingface.co/collections/google/gemma-2-release-667d6600fd5220e7b967f315)    | ✅ ** | [9B](magpie-gemma2-9b.sh),[27B](scripts/magpie-gemma2-27b.sh) | [27B](https://huggingface.co/collections/Magpie-Align/magpie-gemma2-datasets-669da6aff21b09fdcecbd6ea) | 534K |
| [Gemma-1.1](https://huggingface.co/collections/google/gemma-release-65d5efbccdbb8c4202ec078b)    | ⭕️ | [7B](scripts/magpie-gemma7b.sh)
| [Llama 2](https://huggingface.co/collections/meta-llama/llama-2-family-661da1f90a9d678b6f55773b)   | ⭕️ | [7B](scripts/magpie-llama2-7b.sh),[70B](scripts/magpie-llama2-70b.sh)
| [Vicuna](https://lmsys.org/blog/2023-03-30-vicuna/)   | ⭕️ | [7B](scripts/magpie-vicuna-7b.sh)
| [Mistral](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3)   | ⭕️ | [7B](scripts/magpie-mistral7b.sh)
| [Yi](https://huggingface.co/collections/01-ai/yi-15-2024-05-663f3ecab5f815a3eaca7ca8)    | ⭕️ | [34B](scripts/magpie-yi34b.sh)
| [DeepSeek Coder](https://huggingface.co/collections/deepseek-ai/deepseekcoder-v2-666bf4b274a5f556827ceeca) | ⭕️ | [Coder V2 Lite](https://github.com/magpie-align/magpie/blob/main/scripts/magpie-deepseek-coderv2-lite.sh)  

- ✅: It works great! (**\*** Apply a logits processor to surpress markdown; **\*\*** Apply a [filter](exp/str_utils.py) before generating responses.)
- ⭕️: It works! We can get something interesting, but we may need to design an additional logit processor and/or a filter.
- ❌: Not work.
- ❓: Untested.

The navigation of all available Magpie datasets can be found [here](navigation.md).

We hope Magpie can contribute to the democratization of AI with enhanced transparency of model alignment processes!

## Abstract
<details><summary>Click Here</summary>
High-quality instruction data is critical for aligning large language models (LLMs). Although some models, such as Llama-3-Instruct, have open weights, their alignment data remain private, which hinders the democratization of AI. High human labor costs and a limited, predefined scope for prompting prevent existing open-source data creation methods from scaling effectively, potentially limiting the diversity and quality of public alignment datasets. Is it possible to synthesize high-quality instruction data at scale by extracting it directly from an aligned LLM? We present a self-synthesis method for generating large-scale alignment data named Magpie. Our key observation is that aligned LLMs like Llama-3-Instruct can generate a user query when we input only the left-side templates up to the position reserved for user messages, thanks to their auto-regressive nature. We use this method to prompt Llama-3-Instruct and generate 4 million instructions along with their corresponding responses. We perform a comprehensive analysis of the extracted data and select 300K high-quality instances. To compare Magpie data with other public instruction datasets, we fine-tune Llama-3-8B-Base with each dataset and evaluate the performance of the fine-tuned models. Our results indicate that in some tasks, models fine-tuned with Magpie perform comparably to the official Llama-3-8B-Instruct, despite the latter being enhanced with 10 million data points through supervised fine-tuning (SFT) and subsequent feedback learning. We also show that using Magpie solely for SFT can surpass the performance of previous public datasets utilized for both SFT and preference optimization, such as direct preference optimization with UltraFeedback. This advantage is evident on alignment benchmarks such as AlpacaEval, ArenaHard, and WildBench.
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

**Play with Jupyter Notebook**

The toy example can be found in [`demo.ipynb`](demo.ipynb). Have fun! 

<a target="_blank" href="https://colab.research.google.com/github/magpie-align/magpie/blob/main/demo.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>

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
