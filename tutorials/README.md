### How to organise your workspace for the course: general tips

Here we list several tips for better organisation of your workspace and coding infrastructure for the course. We encourage you to follow these tips when working with course materials and/or submitting your final project. Also, we find these tips generally useful for any programming work conducted in the MLT program.

#### Coding on mlt-gpu from your local machine

You can login to mltgpu using Secure Shell. To login type the following in the terminal:

`ssh -p 62266 gusXXXXXX@mltgpu.flov.gu.se`

where `gusXXXXXX` is your username. After the login you will land on the the terminal on mltgpu. Logout from the server by typing

`logout`

To run a python notebook from your browser on mltgpu, choose a port between 8000 and 8999, e.g. 8801. If several are running python notebook on mltgpu at the same time, each user should pick their own port. Then, we need to setup port forwarding to forward the local port 8801 over the ssh tunnel to the port 8801 on the mltgpu:

`ssh -p 62266 -L8801:localhost:8801 gusXXXXXX@mltgpu.flov.gu.se`

Again, you will land on mltgpu terminal. Here you can start jupyter notebook using the port that is being forwarded to your local machine:

`jupyter notebook --port=8801`

Go to your local browser and open http://localhost:8801 (or change the last number to any other port that you are using). You may need to supply a token. In this case, from the mltgpu terminal copy the same URLs with the token, e.g. `http://127.0.0.1:8801/?token=ce5058889de1b13a477c86e09e3f08cf25028386c65040e9`. You can also simply copy the token from this link and insert in the token window.

When you are done, save your progress (CTRL/CMD +S), close the jupyter notebook on mltgpu (CTRL/CMD+C) and logout.

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
