Dataset: Flickr8k

Model: Encoder-Decoder CNN+LSTM

Data and Pre-Trained Models Location: /srv/data/aics/03-image-captioning/data on mlt-gpu

Note: use virtual environment to run the code. You will have to install pytorch, torchvision and some other packages.
No worries, if something does not work and there is a missing package you will get an error message, so just take it from there.

*General steps on how to run the scripts:*

1. Use `create_input_files` in `preproc.py` to prepare your caption data and images. They have to be of specific format and structure.
Captions are processed for the `Karpathy` format, while images are processed and saved in `hdf5` format.

2. You can start training by running `train.py`. Make sure you either pass appropriate arguments in terminal or you edit the defaults of argparse in the python script itself.

3. Jupyter notebook `test.ipynb` is for you to have an interactive demo with the model. It also has a script to visualise attention over the image.

Lycka till!
