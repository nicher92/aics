### Codebase for working with multi-modal transformers (MMTs)

This folder includes information on how to install and use VOLTA framework to use in your AICS projects.
This framework provides a unified codebase for a variety of multi-modal transformers.
One issue with many existing repositories and models is that they are either oudated and not supported anymore,
or there are many wrong dependencies/errors within these repositories.
Thus, it is highly recommended to use VOLTA framework for better reproducibility and transparancy since this code wraps up many existing MMTs in a single code.
But first, please read the paper and check the repository:

*VOLTA paper*: https://direct.mit.edu/tacl/article/doi/10.1162/tacl_a_00408/107279/Multimodal-Pretraining-Unmasked-A-Meta-Analysis

*VOLTA GitHub repository*: https://github.com/e-bug/volta


## How to install VOLTA on mlt-gpu

Follow the steps below, they have been tested on mlt-gpu.
They differ a bit from those in the official repository. Last update: 10/01/2022.

```
git clone --recurse-submodules https://github.com/e-bug/volta.git
python3.6 -m pip install -r requirements.txt 
conda install pytorch=1.4.0 torchvision=0.5 cudatoolkit=10.1 -c pytorch

git clone https://github.com/NVIDIA/apex
cd apex
export CUDA_HOME=/usr/local/cuda-10.1/
python3.6 pip install -v --disable-pip-version-check --no-cache-dir ./

module load cuda/10.1
module load gcc/11.2

python3.6 setup.py develop
```


## How to prepare datasets for VOLTA

The main challenge for you is to prepare your own datasets to use it with VOLTA.
First, as the paper states, all models were pre-trained for the Conceptual Captions dataset (e.g., pairs of weakly related images and captions collected from the web),
then they were fine-tuned for a variety of downstream tasks (e.g., in this case, it means that they were first pre-trained, then trained for other tasks).
Each task has language and vision inputs. They need to be formatted accordingly.
The official repository provides scripts to create datasets *and* links to download preprocessed datasets.
We strongly suggest you look at some of those (for example, MSCOCO) to see how data is organised. It should give inspiration for organising your own dataset.

https://github.com/e-bug/volta/blob/main/data/README.md

## Tips on using for your projects

Think about the task at hand. Example cases:

1. You want to take pre-trained ViLBERT and fine-tune it for the task of predicting the masked word in your own dataset.
Luckily, ViLBERT has been pre-trained for the tasked of Masked Language Modelling on Conceptual Captions, so there must be a code for this.
You need to locate this part of the code, carefully read it and think how you would prepare your dataset to run with this code.

2. You want to extract attention weights from ViLBERT. What is attention? Attention scores are multiplication of keys and queries, so you need to look for them in the code.
One example is `line 260` in `/volta/encoders.py`:
```
tt_attention_scores = torch.matmul(t_query_layer, t_key_layer.transpose(-1, -2))  # [bs, num_heads, seq_len, seq_len]
```
What does it tell us? It says that this line calculates attention of predicted text tokens on input text tokens. Why and how?
Because the transformer sees both text tokens and visual regions and, if it learns to predict the masked token (for example), it will look at
text or/and vision. VOLTA provides functionality to control for the modality that is attended.
If you look at the other lines in this script, you will see that there are also attention weights on other modality.
When you test (!) your model, you will need to save attention weights somewhere explicitly, so that then you can visualise them.

3. More tips will be provided!
