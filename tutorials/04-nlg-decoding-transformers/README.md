### Feature Extraction Tools

Original repository for F-RCNN feature extraction and code to run (in Cafe): https://github.com/peteanderson80/bottom-up-attention
The paper on object feature extraction for image captioning: https://arxiv.org/pdf/1707.07998.pdf

However, we use the model's implementation in PyTorch in Detectron2: https://github.com/facebookresearch/grid-feats-vqa.
There is a sequence of steps to prepare training of the object feature extractor on our servers. Up to date information (21-12-09) is listed below, just follow them. If something does not work or works differently, inform the TAs so that they can update this document.

```
0. git clone https://github.com/facebookresearch/grid-feats-vqa
1. conda environment with python 3.6
2. pip install pytorch==1.3.0
3. pip install torchvision==0.4.1
4. export TORCH_CUDA_ARCH_LIST=7.5\;6.1
5. python -m pip install 'git+https://github.com/facebookresearch/detectron2.git@ffff8acc35ea88ad1cb1806ab0f00b4c1c5dbfd9'
6. pip install fvcore==0.1.1.dev200512
7. pip install cython
8. pip install -U 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'
9. python ~/grid-feats-vqa/train_net.py --num-gpus 4 --config-file ~/grid-feats-vqa/configs/R-50-updn.yaml
```
Above we install particular dependencies and prepare the GPU servers for working with specific settings. It is recommended to run everything in Anaconda (or pip) environment. The last command starts training of your object feature extractor. Training takes around 2-3 days on 4 GPUs, the example model with all log files and different checkpoints are available under ```/srv/data/aics/04-nlg-decoding-transformers/```.
For information on how to use trained model and extract features for your dataset, read the 'Feature Extraction' part in the original repository of the code (https://github.com/facebookresearch/grid-feats-vqa). You will need to change your data to the COCO format (check the original repository for more info).

### NLG: Evaluation and Decoding

Original repository with most of the widely used metrics is available here: https://github.com/tylin/coco-caption.
Pay attention to the 'Setup' section, you will need to install some dependencies for different metrics.

However, the repo above is for Python2.7, the updated one is available here: https://github.com/mtanti/coco-caption (Python3+).

When evaluating your models, you will need to import metric classes from .py files in this repo (e.g., BleuScorer in bleu.py), and also will need to pass candidates (generated texts, outputs) and references (ground-truth texts). The general pipeline for evaluation can be viewed in the transformer code that we discuss a bit below.

Code examples for different decoding methods and evaluation are available here: https://github.com/nilinykh/01-par-gen. The paper that describes the code in the repository is available here: https://aclanthology.org/2020.inlg-1.40/.
In particular, inspect `utils.py` for beam search (BeamNode class + beam search itself). You can track decoding back to the particular line in `generate.py` file (https://github.com/nilinykh/01-par-gen/blob/1d2084ae42cc2d534ba56bf50d320ab64eea4264/model/generate.py#L127). We first initialise scorers (for metrics), and after some encoding of input features and processing, we prepare for decoding (https://github.com/nilinykh/01-par-gen/blob/1d2084ae42cc2d534ba56bf50d320ab64eea4264/model/generate.py#L309). Implementation of all decoding mechanisms is available there. Note that we use decoding during test or validation phase, but in train phase we do everything with teacher-forcing.

### Multi-Modal Transformers

An implementation of the two-stream multi-modal transformer for image captioning: https://github.com/yahoo/object_relation_transformer.
There is also the paper link that describes this architecture in more detail. The benefit of using this particular transformer is the way they encode geometric information. It is not a simple set of bounding box coordinates, sizes, locatoins, but rather some relative function (displacement vector), and such more complex geometric representation helps the model to generate spatial relations much better.

There are other transformers: Vl-BERT, LXMERT, UNITER. They all have GitHub repositories, you should be able to access them.
