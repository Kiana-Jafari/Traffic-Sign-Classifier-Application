### Traffic Sign Classifier Application</br>

<img src='https://github.com/Kiana-Jafari/Traffic-Sign-Classifier-Application/blob/main/Analysis/Safe-System.png' width='250' height='250' align=left></img>

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

The model architecture was first introduced by Pierre Sermanet and Yann LeCun. We will build a modified version of that architecture from scratch. It accepts an input image of shape 32x32, feeds it to the convolution layers each with 108 (5x5) filters, followed by ReLU Nonlinearity, and then Max-Pooling layers to reduce the dimensions. This pattern would repeat two times. Finally, the output of the convolution layers would be fed into two fully connected (FC) layers. The final model will have 579k (GS) / 584k (RGB) parameters. (i.e. if we train the model on grayscale images, vs. RGB images). To learn more about the architecture, please refer to the papers in the <a href='#reference'>*Reference*</a> section. The architecture is shown below: 

<img src='https://github.com/Kiana-Jafari/Traffic-Sign-Classifier-Application/blob/main/Analysis/lenet.png' width='350' height='150'></img>

**6. Accuracy & Loss Curve**

As a final accuracy, the model achieved `95.75%` on the training data, and `97.07%` on the validation data. The Accuracy-loss graph indicates that the model is learning well from the training data and is not overfitting. The test set with an accuracy of about `94.64%` suggests that the model is performing well on new data, although there might be still room for improvement.

<img src='https://github.com/Kiana-Jafari/Traffic-Sign-Classifier-Application/blob/main/Analysis/performance.png' width='500' height='350'></img>

**7. Visualizing Predictions**

The model performance was evaluated by making predictions on the test set. A sample of predictions is shown below. Also, the total number of wrong predictions was `677` out of `12630` images. (approx. an error rate of 0.0536). Please refer to the <a href='https://github.com/Kiana-Jafari/Traffic-Sign-Classifier-Application/blob/main/Analysis/preds.png'>predictions</a> section to see the full predcitions.

<img src='https://github.com/Kiana-Jafari/Traffic-Sign-Classifier-Application/blob/main/Analysis/pred_sample.png' width='360' height='360'></img>

**8. Model Performance**

A (normalized) confusion matrix was plotted to define the model performance on the test data. The main evaluation metric used to describe the network performance was **Recall**, which measures how often an algorithm correctly identifies positive instances (true positives) from all the actual positive samples in the dataset.

<img src='https://github.com/Kiana-Jafari/Traffic-Sign-Classifier-Application/blob/main/Analysis/cm.png' width='800' height='600'></img>

Further improvements:
- Add more `Dropout` layers
- Generate more samples for classes with low accuracy (such as "Pedestrian")
- Try to train the model on the traditional machine learning algorithms

**9. Run the model on Streamlit (Application)**

The model was tested on Streamlit. You can experiment with images from the <a href='https://github.com/Kiana-Jafari/Traffic-Sign-Classifier-Application/tree/main/Test%20Images'>Test Images</a> folder. Please note that some images were downloaded from the internet to further test the model's capabilities, and are not representative of the original test data. Therefore, they can't be used for data-driven decision-making. 

To test the application on your local, run the following commands on your Terminal. Please note that you need to install Python dependencies, which can be found in <a href='https://github.com/Kiana-Jafari/Traffic-Sign-Classifier-Application/blob/main/requirements.txt'>this text</a> file.

<pre>pip install streamlit</pre>
<pre>streamlit run app.py</pre>

<img src='https://github.com/Kiana-Jafari/Traffic-Sign-Classifier-Application/blob/main/Analysis/test.png' width='800' height='350'></img>

Feel free to test the model and fine-tune the hyperparameters for better results!

<h3 id='reference'>10. References</h3>

Check the two articles below to find out more information:

- <a href='https://ieeexplore.ieee.org/document/6033589'>Traffic sign recognition with multi-scale Convolutional Networks</a>
- <a href='https://odr.chalmers.se/server/api/core/bitstreams/fdef1142-92cb-4f8c-9c8a-f17f72260c00/content'>Traffic sign classification with deep convolutional neural networks</a>
