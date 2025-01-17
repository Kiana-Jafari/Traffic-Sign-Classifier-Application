### Traffic Sign Classifier Application</br>

<img src='https://github.com/Kiana-Jafari/Traffic-Sign-Classifier-Application/blob/main/Analysis/Safe-System.png' width='360' height='360' align=left></img>

**1. Introduction**

Welcome to this project! Traffic Sign Detection is an important concept in enhancing road safety, autonomous vehicles, and driver-assistance systems. Businesses such as transportation safety organizations often work on improving road safety, preventing crashes, and minimizing their impact by assessing road safety from the perspective of all users, including drivers, cyclists, pedestrians, and motorcyclists. In this project, we implemented a convolutional neural network from scratch to automatically classify traffic signs into predefined categories and deployed the model using Streamlit. Please note that the notebook has a research-educational, yet industrial purpose. The aim is to build a convolutional neural network from scratch, train it on a slightly different version of the original dataset and the architecture, and replicate the author's work.</br>

___________________________________________________________________________________________________________________________________________________________________________________________

**2. Dataset**

We're going to use The **German Traffic Sign Recognition Benchmark (GTSRB)**. It contains 43 classes of traffic signs and is split into 39,209 training images and 12,630 test images. You can find the datasets in the <a href='https://github.com/Kiana-Jafari/Traffic-Sign-Classifier-Application/tree/main/Data'>Data</a> folder. Since the sets are too big, they've been split into smaller chunks of sub-files. Please note that you can also *Clone* the repository containing the dataset into the google colab. This will allow access to the dataset inside the colab environment:

<pre>!git clone https://bitbucket.org/jadslim/german-traffic-signs</pre>

**3. A sample of images**

Traffic sign examples in the GTSRB dataset were extracted from 1-second video sequences. They have varying light conditions and rich backgrounds. A sample of images is shown below:

<img src='https://github.com/Kiana-Jafari/Traffic-Sign-Classifier-Application/blob/main/Analysis/sample.png' width='360' height='360'></img>

**4. Distribution & Data Augmentation**

The first histogram shows the distribution of classes (see figure). It clearly indicates that we have *class imbalance*. For this reason, **Image Augmentation** technique is applied to the images.

<img src='https://github.com/Kiana-Jafari/Traffic-Sign-Classifier-Application/blob/main/Analysis/hist.png' width='400' height='250'></img>

Other Data Preprocessing techniques used in this project:
- Converting RGB images to Grayscale
- Histogram Equalization
- Normalization

**5. Model Architecture**

The model architecture was first introduced by Pierre Sermanet and Yann LeCun. We will build a modified version of that architecture from scratch. It accepts an input image of shape 32x32, feeds it to the convolution layers each with 108 (5x5) filters, followed by ReLU Nonlinearity, and then Max-Pooling layers to reduce the dimensions. This pattern would repeat two times. Finally, the output of the convolution layers would be fed into two fully connected (FC) layers. The final model will have 579k (GS) / 584k (RGB) parameters. (i.e. if we train the model on grayscale images, vs. RGB images). To learn more about the architecture, please refer to the papers in the <a href=''>*Reference*</a> section. The architecture is shown below: 

<img src='https://github.com/Kiana-Jafari/Traffic-Sign-Classifier-Application/blob/main/Analysis/lenet.png' width='350' height='150'></img>

**6. Accuracy & Loss Curve**

As a final accuracy, the model achieved `95.75%` on the training data, and `97.07%` on the validation data. Also, the accuracy on the test set was about `94.64%`. The Accuracy-loss graph is shown below: 

<img src='https://github.com/Kiana-Jafari/Traffic-Sign-Classifier-Application/blob/main/Analysis/performance.png' width='500' height='330'></img>
