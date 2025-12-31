## Installation

```shell
pip3 install -U pip
conda create -n plm python=3.10
conda activate plm
pip3 install torch torchvision torchaudio
conda install transformers
```

## Model downloading

We use DeepSeek-AI's [DeepSeek-R1-Distill-Qwen-7/14/32B](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B) and Qwen's [Qwen/Qwen2.5-7/14/32B-Instruct](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct) as the main architecture.

Please also download the above models for experiments.


## Model Evaluation
```
bash run.sh
```