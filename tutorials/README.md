### How to organise your workspace for the course: general tips

Here we list several tips for better organisation of your workspace and coding infrastructure for the course. We encourage you to follow these tips when working with course materials and/or submitting your final project. Also, we find these tips generally useful for any programming work conducted in the MLT program.

#### Coding on mlt-gpu from your local machine
You can run your scripts on mlt-gpu by running everything on your local computer. First, enter the following command in the terminal on your local machine: ``ssh -p 62266 [yourName]@mltgpu.flov.gu.se -L8000:localhost:8000``, where 'yourName' is your gu-account. This should connect you to mlt-gpu and book a port (8000) on the server for you. The port can have any number, which someone does not already use. Then, on your local machine, open ``127.0.0.1:8000`` in your browser. You should see the starting window for Jupyter Lab (given that you installed it on mlt-gpu). Then, if required, enter the login token; you can access it in the terminal where you ran your ssh command. That's it! Now you can navigate mlt-gpu from your local machine and run any code remotely.  
More detailed information is available on Canvas course webpage under Pages/Course workspace: mlt-gpu, Jupyter notebooks.

#### Use virtual environments
Virtual environments allow you to avoid conflicts between different versions of your project and help your TAs navigate your code and test it. For more, you can read [this thread](https://stackoverflow.com/questions/41972261/what-is-a-virtualenv-and-why-should-i-use-one) (installing environments with pip) or [this one](https://towardsdatascience.com/introduction-to-conda-virtual-environments-eaea4ac84e28) (installing environments with conda).

#### Use Jupyter Lab for interactive work
Use Jupyter Lab when you want to inspect individual parts of the code and/or want to 'interact' more with your code. Python scripts (.py) should be used when you submit your project, but an interactive notebook should accompany script testing, discussions with TAs regarding the code. It is generally more convenient to work this way since your TA (and you) could run individual parts of the code in isolation when required.

#### Use screen/tmux
You often want to train your model overnight without breaking/stopping the training scripts. To achieve this, use [screen](https://linuxize.com/post/how-to-use-linux-screen/) or [tmux](https://linuxize.com/post/getting-started-with-tmux/). They are already installed on mlt-gpu and accessible to everyone. Start your scripts in the screen/tmux session (you have to create one first), and your code will run safely in the background even when you log out. Beforehand, we suggest you test your scripts on a small subset of your data to ensure that training, validation, evaluation, and testing loops are working well. Otherwise, when you check your screen/tmux session the next day, you can see an unexpected error coming from your code, which means you will have to re-run your scripts.

#### Where to store data / find data / set up permissions
All data used in the course tutorials can be found in ``/srv/data/aics``. There you can also find data from previous course iterations. When working with your own data, please store it in ``/srv/data/yourName``, where 'yourName' is the folder you create yourself. This way, your TAs will access your models/checkpoints/large files when looking at your project. Finally, all scripts and your code related to your course work should be uploaded to GitHub in a publicly accessible repository (you also need to create one). This way, TAs will be able to test both your code (from the GitHub repo) and run it with the models (from /srv/data/).
Always check if the dataset you work with is found somewhere on mlt-gpu, because it could have been used by other students or by your TAs. This way, you will make sure that you do not occupy extra space with yet another copy of the same dataset.

#### CUDA-related details (to be updated)
Make sure that you set up a particular GPU for your scripts by putting, for example, the following lines in your code:  

```
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID". 
DEVICE = torch.device('cuda:INTEGER')
```

, where INTEGER is id of the gpu you want to use.

A different method is to explicitly restrict your scripts to run on a specific gpu. Enter the following command in your terminal:  

```
export CUDA_VISIBLE_DEVICES=INTEGER
```

#### Project Submission
By the end of the course, you are required to submit a project to get course credits. The submission consists of (1) a paper and (2) a public GitHub repository that accompanies your paper. Typically, you first run your code on mlt-gpu when working on it (e.g., as notebooks /.py scripts), and then you upload it to your GitHub repo when it's about time to submit it (in the form of .py files and documentation in README.md). The project paper should be approximately 6-8 pages, but it depends on each case individually. For the project paper, we encourage you to use LaTeX, perhaps, [this link](https://www.overleaf.com/learn/latex/Tutorials) can be helpful for you to start with. Also, you should use official ACL latex templates for your project (check it [here](https://www.overleaf.com/latex/templates/acl-2020-proceedings-template/zsrkcwjptpcd)) or official GU template (which you will also probably use for your thesis).
