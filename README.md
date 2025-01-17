### Traffic Sign Classifier Application</br>

<img src='https://github.com/Kiana-Jafari/Traffic-Sign-Classifier-Application/blob/main/Analysis/Safe-System.png' width='360' height='360' align=left></img>

####**1. Introduction**

Welcome to this project! Traffic Sign Detection is an important concept in enhancing road safety, autonomous vehicles, and driver-assistance systems. Businesses such as transportation safety organizations often work on improving road safety, preventing crashes, and minimizing their impact by assessing road safety from the perspective of all users, including drivers, cyclists, pedestrians, and motorcyclists. In this project, we implemented a convolutional neural network from scratch to automatically classify traffic signs into predefined categories and deployed the model using Streamlit. Please note that the notebook has a research-educational, yet industrial purpose. The aim is to build a convolutional neural network from scratch, train it on a slightly different version of the original dataset and the architecture, and replicate the author's work.</br>

___________________________________________________________________________________________________________________________________________________________________________________________

**2. Dataset**

We're going to use The **German Traffic Sign Recognition Benchmark (GTSRB)**, which contains 43 classes of traffic signs, split into 39,209 training images and 12,630 test images. You can find the datasets in the `Data` folder. Since the files are too big, they've been split into smaller chunks of sub-files. Please note that you can also *Clone* the repository containing the dataset into the google colab. This will allow access to the dataset inside the colab environment:

<pre>!git clone https://bitbucket.org/jadslim/german-traffic-signs</pre>

**3. A sample of images**
Traffic sign examples in the GTSRB dataset were extracted from 1-second video sequences. They also have varying light conditions and rich backgrounds. A sample of images is shown below:

<img src='https://github.com/Kiana-Jafari/Traffic-Sign-Classifier-Application/blob/main/Analysis/sample.png' width='300' height='300'></img>
