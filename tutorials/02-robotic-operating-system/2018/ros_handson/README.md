# ROS Handson
A quick tutorial for ROS. 

The introductory slides for ROS is available here:

* Nov 2018 https://docs.google.com/presentation/d/19Jux1hVIHitNU_VemnwrXorucI4A1l9PiAs1C_yQmpY
* Nov 2017 https://docs.google.com/presentation/d/1IMnimDSfmhllHSQcyfhACHd52AuHpX0Tv_qRDCdptJ0

# Requirements

* Ubuntu 16.04 
* ROS Kinetik

You can install latest version of [Virtual Box](https://www.virtualbox.org/wiki/Downloads) and its extesion pack, then download/run this Ubuntu machine with ROS:
https://gubox.box.com/s/spdikbi252bcrgy3qbwn963y0vqjfzqw

(additional intorduction for kinect)
* Standalone Ubuntu machine (not a virtual machine) 16.04
* Kinect/RGBD device
* freenect driver on Ubuntu and ROS `sudo apt-get install freenect ros-kinetic-freenect-stack`


# Step-by-step 

Follow these steps:

### 1. Manage the shell environment

Run this:

```
printenv | grep ROS
```

The result must be something like this:

```
ROS_ROOT=/opt/ros/kinetic/share/ros
ROS_PACKAGE_PATH=/opt/ros/kinetic/share
ROS_MASTER_URI=http://localhost:11311
ROSLISP_PACKAGE_DIRECTORIES=
ROS_DISTRO=kinetic
ROS_ETC_DIR=/opt/ros/kinetic/etc/ros
```

Run `source /opt/ros/kinetic/setup.bash` if the environment was not ready.


### 2. Create a Workspace
'catkin' is a build system based on CMake and extended with Python.
most famouse IDEs have their own build system to maintain processes 
related to maintainig packages in a project and run them for test.
catkin prepare dependencies and the code to be build and run for ros.
This is how you can create a catkin workspace which you can write python 
codes for ros:
```
mkdir -p ~/ros_ws/src
cd ~/ros_ws/
catkin_make
```

Now that the folder is ready to be used as a ros workspace, you must source
this workspace for shell environment:
```
source devel/setup.bash
```

In order to check if this is correct:
```
echo $ROS_PACKAGE_PATH
```

The workspace forlder is part of the `ROS_PACKAGE_PATH` for example
`/home/<username>/ros_ws/src:/opt/ros/kinetic/share`

In order to understand the ROS file system and folders follow [the ROS tutorial](http://wiki.ros.org/ROS/Tutorials/NavigatingTheFilesystem). 

### 3. Create and build a package

Within the workspace for your projects, you can have several packages. 
each package is a folder containing the minimum specifications of its 
CMake build and the required packages.

Using the command line tool `catkin_create_pkg` is an easy way to initialize
a new package: `catkin_create_pkg <package_name> [depend1] [depend2] [depend3]`
As an example, we create a package for this tutorial as following:
```
cd ~/ros_ws/src
catkin_create_pkg this_tutorial std_msgs rospy
```

Now you need to rebuild the workspace, and source it again:
```
cd ~/ros_ws/
catkin_make
. ~/ros_ws/devel/setup.bash
```

For more details you can read the documentations and follow more advanced
tutorials in the [wiki.ros.org](wiki.ros.org).

### 4. Running ROS and your programs in ROS
ROS needs its core program runing in background. The easiest way to run it is as following:
```
roscore
```

`roscore` starts the master process with one node. We can see list of nodes 
with the following command:
```
rosnode list
```

*HINTS: you can run commands with nohup in background: `nohup roscore &`
but in this tutorial, it is easier to open each process in separate terminals*

The next step is to run a python code in ros.
here we show how to write a python code in the package folder and run it.
then we will expand this small program to become a node in ROS.

First, let's run a hello world for ros:
```
echo '#!/usr/bin/env python
print("hello rosrun!")
' > ~/ros_ws/src/this_tutorial/src/hello.py
```

It needs to be executable as well: 
```
chmod +x ~/ros_ws/src/this_tutorial/src/hello.py
```

This is how you can run it:
```
rosrun this_tutorial hello.py
```

*HINT: you can kill nodes with `ctrl+c`*

Next step is to make it persistent.
In order make it a valid ros node we just need to assign it a name,
and tell the ros master.
```
echo '#!/usr/bin/env python
import rospy
rospy.init_node("greeter")
rate = rospy.Rate(1) # it means 1hz
while not rospy.is_shutdown():
   print("hello world {0}".format(rospy.get_time()))
   rate.sleep()
' > ~/ros_ws/src/this_tutorial/src/hello.py
```

Now, if you run it again it stays open:
```
rosrun this_tutorial hello.py
```

You can check its info, in another terminal:
```
rosnode info \greeter
```

Nodes alone cannot communicate with eachother,
messages must be communicated under a ros topic.
you can list all topics in command line:
```
rostopic list
```

We can start a new topic and send signals there:
```
echo '#!/usr/bin/env python
import rospy
from std_msgs.msg import String
rospy.init_node("greeter")
rate = rospy.Rate(1) # it means 1hz
pub = rospy.Publisher("greeting_topic", String, queue_size=1)
while not rospy.is_shutdown():
   pub.publish("hello world {0}".format(rospy.get_time()))
   rate.sleep()
' > ~/ros_ws/src/this_tutorial/src/hello.py
```

Now, when we run this node there is no sign of input in this terminal.
```
rosrun this_tutorial hello.py
```

In a seperate terminal, we can check if the new topic is active.
```
rostopic list
```

Then, if the /greeting_topic is in the list, we can get the detailed
information about the publisher and its datatype:
```
rostopic info /greeting_topic
```

We can echo the content that are published in the topic:
```
rostopic echo /greeting_topic
```

Similar to the echo action for rostopic, we can also write a node
which process the published content and then publishes a new materrial.
```
echo '#!/usr/bin/env python
import rospy
from std_msgs.msg import String

def callback(string):
   pub.publish("I heared \"{0}\"".format(string.data))

rospy.init_node("feedbacker")
pub = rospy.Publisher("feedback_topic", String, queue_size=1)
sub = rospy.Subscriber("greeting_topic", String, callback)
rospy.spin()
' > ~/ros_ws/src/this_tutorial/src/feedback.py
chmod +x ~/ros_ws/src/this_tutorial/src/feedback.py
```

Notice that we don't need infinit loop to keep the node running now. And the frequency of publishing content in feedback_topic depends on the how frequent the subscriber calls the callback function.


Now, you can run both of them in two terminals:

**Terminal 1:**
```
. ~/ros_ws/devel/setup.bash
rosrun this_tutorial hello.py
```

**Terminal 2:**
```
. ~/ros_ws/devel/setup.bash
rosrun this_tutorial feedback.py
```

### 5. Record topics

We often need to record all topics at once to be able to replicate them later. ROS provide this with `rosbag`. First, you can create a folder for saving it on a file:

```
mkdir ~/bagfiles
```

While two nodes are running, we can open another terminal and record them all in one file:

```
cd ~/bagfiles
rosbag record -a
```

Later, you can just play the nodes without running them:

```
rosbag play <your bagfile>
```

### 6. Explore ROS tutorials

It is very common to have several nodes running which depend on each other at the same time. The command for running them at once can be packed into a `roslaunch` script. You can read about this among other related topics in [ROS tutorials](http://wiki.ros.org/ROS/Tutorials). 



# Understanding Kinect and Keras with ROS

### Working with Kinect

You can test if Kinect is connected and working on Ubuntu by running the test commands:

```
freenect-glview
```

You can also test other features such as its motor:

```
freenect-tiltdemo
```
In order to read from Kinect sensors in ROS, you need to run all topics of sensors with the following command:

```
roslaunch freenect_launch freenect.launch
```

Then in another terminal, you can check if all topics are working:

Just for listing them:
```
rostopic list
```

Try to read the data:
```
rostopic echo /camera/rgb/image_color
```

In order to visualise them you can use `rviz` to map them in 3D space. Open `rviz` with the following command:

```
rosrun rviz rviz
```

Now, include the point clouds and set the related channels to depth and color images.

### Running Keras

We can improve the greeter node from previous section with image recognition. Download `keras-application.py` and try to understand the code. This node requires images from camera. You can use Kinect camera with `freenect_launch`. 
Put the file in `~/ros_ws/src/this_tutorial/src/keras-application.py` and run it:

```
rosrun this_tutorial keras-application.py
```

This probably propts an error. In order to fix this problem, you can download the dummy elephant image to start the keras sessions. You can find the dummy file in `src/elephant.jpg`. Npow, try again. OpenCV is going to open a window from camera content. Pressing `r` will run the recognition process and `q` exits the window. In another terminal you can read the published messages as before:

```
rosrun this_tutorial feedback.py
```




