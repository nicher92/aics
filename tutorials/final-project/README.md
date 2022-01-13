# Codebase for working with multi-modal transformers (MMTs)

This folder includes information on how to install and use VOLTA framework to use in your AICS projects.
This framework provides a unified codebase for a variety of multi-modal transformers.
One issue with many existing repositories and models is that they are either oudated and not supported anymore,
or there are many wrong dependencies/errors within these repositories.
Thus, it is highly recommended to use VOLTA framework for better reproducibility and transparancy since this code wraps up many existing MMTs in a single code.
But first, please read the paper and check the repository:

*VOLTA paper*: https://direct.mit.edu/tacl/article/doi/10.1162/tacl_a_00408/107279/Multimodal-Pretraining-Unmasked-A-Meta-Analysis

*VOLTA GitHub repository*: https://github.com/e-bug/volta


## How to install VOLTA on mlt-gpu

Follow the steps below, I have tested the installation with training and testing for MSCOCO retrieval task.
The steps differ from those in the official repository, please install the framework the way it is described below.
Last update: 13/01/2022.

First, clone the repository with all submodules in it:

```
git clone --recurse-submodules https://github.com/e-bug/volta.git
```

Then, create a conda environment and install all dependencies. Note that I ran pip-related installations by explicitly calling `python3.6`.
There could be multiple versions of Python installed in your home directory: internal (system) Python, versions from other environments.
When you run `python` in your installed environment, you expect it to use `python3.6`, but what often happens is that conda is using some other version of Python.
You can either add path to your conda-based Python 3.6 to your PATH environment variable (or, simply put the python path in `bashrc` file, and it will be sourced automatically every time you log in), or explicitly run every command by calling desired python (as I do below). Read a little more about environment variables and files like `bashrc` [here](https://bic-berkeley.github.io/psych-214-fall-2016/using_pythonpath.html).

```
conda create -n volta python=3.6
conda activate volta
python3.6 -m pip install -r requirements.txt
```

Next, we are going to install Pytorch (specific version of it). It is important that you install this version, because newer versions
might cause dependency conflicts later (conflicts with CUDA, gcc, etc.).

```
conda install pytorch=1.4.0 torchvision=0.5 cudatoolkit=10.1 -c pytorch
```

Now, we want to install Apex. This library allows us to perform distributed training (e.g., allow GPUs to distribute the job between themselves in order to ease training, make it faster and more precise). There are many benefits to it, you can read more [here](https://developer.nvidia.com/blog/apex-pytorch-easy-mixed-precision-training/).
Installing Apex is not that simple, below I explain the steps you need to take to do it on mlt-gpu.
First, clone the corresponding GitHub repository in your home directory. We also need to set CUDA_HOME to the CUDA version we want to use (mlt-gpu has several CUDA versions, you often need to choose a specific one):

```
git clone https://github.com/NVIDIA/apex
export CUDA_HOME=/usr/local/cuda-10.1/
```

Now, Apex will use our GPUs (e.g., CUDA) and since we installed CUDA 10.1, we need to make sure that versions of all other packages are correct.
In particular, the problem occurs with innate gcc version which is currently too new for CUDA 10.1 on mlt-gpu.
We will automatically set it to the one that fits us:

```
module load cuda/10.1
```
You should see a message that will mention what has been reloaded with a version change. In particular, gcc should be downgraded to `gcc/8.3` or lower, but its version should not be higher than 8.

Next, gcc has some library dependencies which are outdated for the current Ubuntu version that we use on mlt-gpu. Specifically, you will encounter the error while training your models that will say that gcc cannot locate `libmpfr.so.6`. Through trial and error I have discovered that this version needs `libmpfr.so.4` instead. But our Ubuntu has removed this library because it is quite oudated. And as normal users we do not have root rights to install any package of any version on mlt-gpu. Thus, we are going to download the file we need and point its location in the `LD_LIBRARY_PATH`, an environment variable that is used by the system to locate packages. We do the steps below in our home directory. `user_dir` stands for your home directory on mlt-gpu.

```
wget https://repo.almalinux.org/almalinux/8/BaseOS/x86_64/os/Packages/mpfr-3.1.6-1.el8.x86_64.rpm
rpm2cpio mpfr-3.1.6-1.el8.x86_64.rpm | cpio -idmv
export LD_LIBRARY_PATH=/home/{user_dir}/usr/lib64:$LD_LIBRARY_PATH
```

We are nearly there to compile Apex. Still, there will be a problem with Apex not finding some files, because Pytorch version is not the most recent one. Specifically, since we are using transformers, Apex will need to look for a file that allows distributed training with multi-head self-attention, and this file is available with Pytorch 1.8 or higher. So we are going to reset Apex repository version to an earlier version and install it from there:

```
cd apex
git reset --hard 3fe10b5597ba14a748ebb271a6ab97c09c5701ac
python3.6 setup.py install --cpp_ext --cuda_ext.
```

Phew, no error should occur anymore. Now, just complete the rest of the steps to install Volta:

```
cd ~/volta
cd tools/refer; make
python3.6 setup.py develop
```

## Before you start using Volta

With this setup I was able to train and test ViLBERT on MSCOCO retrieval task without problems. I took ViLBERT pre-trained on Conceptual Captions and fine-tuned it for the MSCOCO retrieval task. One epoch on average took 3 hours. When you fine-tune it for your downstream task, I recommend you to do it for 3-5 epochs, doing it for 20 epochs (as in Volta) will take too much time and space.

Also, before running train.sh or test.sh, please set your GPUs by running the command below. I ran my code on three GPUs, but you might choose one of those or two or all four. Check which GPU has space avavilable (run `nvidia-smi`) and set the IDs below accordingly.

```
export CUDA_VISIBLE_DEVICES=1,2,3
```

Please respect time of other students, because there are many projects and many training processes which will occupy space. Do not be hungry for memory space and let others also train/test their things. If you see that someone is using too much space, you can identify this gu-account by running the following:

```
ps -up `nvidia-smi -q -x | grep pid | sed -e 's/<pid>//g' -e 's/<\/pid>//g' -e 's/^[[:space:]]*//'`
```

This will print full output of which processes are running on GPUs and who are the users that use them. Talk to these persons and decide how to distribute space between your projects.

Important: `train_task.py` does not have `sys` imported, so please add `import sys` in the beginning of the file. This package is required when you validate your model. If you experience any other error or problem (e.g., missing import), please let me know and we will update ths file.


## How to prepare datasets for VOLTA

The main challenge for you is to prepare your own datasets to use it with VOLTA.
First, as the paper states, all models were pre-trained for the Conceptual Captions dataset (e.g., pairs of weakly related images and captions collected from the web),
then they were fine-tuned for a variety of downstream tasks (e.g., in this case, it means that they were first pre-trained, then trained for other tasks).
Each task has language and vision inputs. They need to be formatted accordingly.
The official repository provides scripts to create datasets *and* links to download preprocessed datasets.
We strongly suggest you look at some of those (for example, MSCOCO) to see how data is organised. It should give inspiration for organising your own dataset.

https://github.com/e-bug/volta/blob/main/data/README.md

## Tips on using VOLTA for your projects

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

3. Multi-modal transformers are often using object detections. The detector that we have is based on the [this repository](https://github.com/facebookresearch/grid-feats-vqa/tree/2d96ab01ed74cb1e1dc06401ceda8daa3e14f5db), and it extracts from 10 to 100 objects for image image. When you get these objects (contact me to extract them on your data), you will need to think how to set the number of objects to a single value. For example, you want all of your images to have 36 objects. You can either pad zeros or remove objects, for which the extractor is not that confident (`cls_prob` output of the extractor contains these confidence scores for each object). Choosing which method is the best is totally up to you: just explain your logic behind it very clearly in the project paper.

4. Do not forget to update paths in bash scripts *and* in model's configuration files found in `volta/config_tasks`.

5. You will also need to update class `ImageFeaturesH5Reader`, which is loading object features for the modal. At the moment, Volta is expecting `.h5` files with features, but our extractor provides `.npy` file with features, bounding boxes, class labels and class confidence scores for each image separately. You will need to add a for-loop or similar and edit the code to be able to use `.npy` files.

6. FYI: the pre-trained checkpoints in the CTRL setting (available [here](https://github.com/e-bug/volta/blob/main/MODELS.md)) are fine-tuned models for the given tasks. Think if you want to fine-tune the original model pre-trained on Conceptual Captions on your dataset/task *OR* you want to use a fine-tuned version which was additionally trained on several downstream tasks. In a way, your own task is a fine-tuning task, so there is a benefit to take the pre-trained model. But, perhaps, using fine-tuned checkpoint released by the authors will also bring a benefit? It is up to you to decide and argue about it in your project report.


## Useful links

1. https://github.com/NVIDIA/apex/issues/156#
2. https://github.com/NVIDIA/apex/issues/1043
3. https://stackoverflow.com/questions/5905434/building-gcc-4-6-libmpfr-so-4-cannot-open-shared-object-file
4. https://stackoverflow.com/questions/51288467/cannot-open-libmpfr-so-4-after-update-on-ubuntu-18-04
