Image Classification Tutorial
models.py describes the CNN model used to process images, extract their features and classify into num_classes number of classes.

dataset.py is a script that is necessary to pre-process data for training, validation and testing. If you want to work with the raw data, it can be found on mlt-gpu under /srv/data/aics/image-classification/2021/.

config.json is a standard file with hyperparameters. It is nice to have this file separately so that you can simply change information in this file only, if you want to re-train your model with different settings.

train_df.csv is the training dataset of images and their ground-truth labels. This is an example of how to prepare and organise data for model training.

train.py is a very important script that trains and validates your model. Make sure you correctly adjust the path where your model will be saved. You can also write a couple of lines to force saving of the model only after particular epochs - this is an idea, it is not in the code, but it is something that can be done.

test.ipynb is an interactive script that uses a pre-trained model to classify and display some examples of images and predicted labels. To run it, you would need to have access to data (e.g., download data and edit paths) on mlt-gpu (the folder mentioned above).

Generally, try to organise your code in .py files. use .ipynb files only when you want to show something interactive, most likely during the class. In your projects, you are encouraged to submit a bunch of .py files with similar hierarchy that you see in this folder.
