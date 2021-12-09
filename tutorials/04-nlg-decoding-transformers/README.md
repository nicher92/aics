### Feature Extraction Tools

Original repository for F-RCNN feature extraction and code to run (in Cafe): https://github.com/peteanderson80/bottom-up-attention
The paper on object feature extraction for image captioning: https://arxiv.org/pdf/1707.07998.pdf

However, we use the model's implementation in PyTorch in Detectron2: https://github.com/facebookresearch/grid-feats-vqa.
There is a sequence of steps to prepare training of the object feature extractor on our servers. Up to date information (21-12-09) is listed below, just follow the steps:

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
