# Exercise 10 - Real Time Object Detection Using Viola Jones Method
---
In this exercise, we take a look at the use case of the widely deployed Viola-Jones algorithm - a method that enables real-time face detection. Object detection is a core problem within computer vision, and with its introduction in 2001, Viola Jones became the first face detection algorithm used in real-time. Even for its ancient beginnings, it is still applicable today due to relatively high accuracy performance in conjunction with its low compute requirement, which is still competitive with much later frameworks such as YOLO v3. The algorithm has later been integrated within large applications such as *Snapchat*, which use the framework within their popular *Snapchat Lenses* widget. Later on, we'll explore how face detection can be adopted for such uses. Similarly we'll show, that the algorithm can be adapted to detect other object classes.

![[Pasted image 20251118110131.png|600]]

### Learning objectives
---
After completing this exercise, the student should be able to do the following:
1. Be able to use opencv pretrained cv::CascadeClassifier()
2. Count the number of detections from a pretrained cv::CascadeClassifier
3. Compute outputs of haar features using an integral image
4. Interpret how simple Haar-features relate to common properties of the class to be detected
5. Identify failure modes of Viola-Jones with respect to simple image conditions such as lighting, object transformations, distance and resolution
6. Relate the scale of a Viola-Jones object detector to a specific task, considering object sizes

## Training a Cascade Classifier
---
In the following exercises, we'll investigate the performance of the Viola-Jones algorithm when trained from scratch on a small dataset. We'll show, that the framework doesn't necessarily need to be used for faces, but is a general object detection tool which can be trained to solve any object detection problem.

We'll start by employing a pretrained stop-sign object detector for image capture. To do this, we'll use the script **VJ_performance_eval.py**. You can find all the scripts and data we'll need [here](linkidinky.com). Download the folder and *cd* into it.
### Exercise N
In your terminal, run the following command:

```
python VJ_performance_eval.py --model stop_sign_detector.xml --input stop_sign.jpg --type image --w 24 --h 24
```

**Question N**: *Does the model perform as expected?*

### Exercise N+1
In order to understand the underlying decision logic that the model makes during execution, we'll take a look at the computed Haar features of the model at different stages. Refer to the directory **haar_features** to get insights into what the model has learned to put attention towards at different stages.

**Question N+1**: *Looking at some of the computed haar features, can you tell what the model attends to at the different classification stages?*

Now choose a specific feature image that you find interesting. We'd now like to investigate what the model computes at this stage.

**Question N+2**: *Load image and find slice*

**Question N+3**: *Visualize 24x24 matrix. Can you extract any information from the matrix?*

**Question N+4**: *Compute the Integral Image from (0,0). From the integral image, compute the output of the chosen haar feature.*

### Exercise N+2
Now we'd like to look at the performance of the stop sign detector when tasked with a moving image, to see whether it is robust to real-time image changes, such as object scaling and pose adjustments. Again run the file **VJ_performance_eval.py**, now using the videos **van_video.mp4** and **scale_diff.mp4** as image capture input. Remember to change the type of input to video format.

**Question N+5**: *How does the model perform? Can you find any weaknesses that it possesses?*

During detection, the model computes features based on a series of scaled image sizes, forming a pyramid of images. OpenCV allows us to modify this pyramid through their detection framework.

**Question N+6**: *Examine the script file and try to modify the scale_factor parameter of the detection function. Explain the effects of increasing and decreasing the scale factor.*

## Training an object detector (Optional)
---
The stop-sign model was trained using data from Caltech-101, consisting of 64 training images with bounding box annotations. We'll now try to train our own object detector from the same dataset! The dataset contains 101 annotated object categories each with separate subset sizes, ranging from 40-800 images. You can download the dataset from [here](https://data.caltech.edu/records/mzrjq-6wc02). In order for everything to work as expected, you need to place the root of the Caltech-101 dataset into the **data** folder.

In order to perform this exercise, we highly recommend using the Anaconda Python distribution. Training an object detector using tools provided by OpenCV requires an older version of OpenCV, which can be handled much easier using Anaconda.

### Exercise N+3 (Optional)
We first need to setup a new virtual environment, in order to run the tools we need, which have been disabled for newer versions of OpenCV (>3.4) . To start with, if you're currently in a virtual environment, deactivate by running:

```
conda deactivate
```

Now we create a virtual environment with the necessary installation of OpenCV and activate it. This can be done as follows:

```
conda create -n obj_train python 'opencv>=3,<4'
conda activate obj_train
```

### Exercise N+4 (Optional)
In order for our data to fit to the semantic notation needed for the OpenCV training tools, we first need to perform some pre-processing on our annotations. To do so, you need to choose **1)** a *positive* class, denoting the class you want your model to identify; and **2)** a *negative* class, denoting a mock class intended to act as a model *regularizer*, meaning that we try to nudge our model to learn relevant features instead of erroneous features. In the case of the stop-sign model, the chosen positive class was *stop_sign* and the negative class was *car_side*.

Run the following command when at the root of the **process_dataset.py** script:

```
python process_dataset.py --pos <pos_class> --neg <neg_class>
```

With *pos_class* and *neg_class* being the explicit names of the categories in the Caltech-101 dataset.

Having ran this script, the description files *pos_class_pos.dat* and *pos_class_neg.dat* should be contained in the Caltech-101 folder along with their gray-scaled datasets in the folders **pos_class_gray** and **neg_class_gray**.

### Exercise N+5 (Optional)
Now we'll commence the training portion of the exercise! To start with, *cd* into the Caltech-101 folder, as the following OpenCV tools only work from the folder where the data resides. We need to create a *positive vector file*, which provides the path from the positive images to the positive description file. This can be done as follows:

```
opencv_createsamples -info <pos_class>_pos.dat -vec positive.vec -w <width> -h <height>
```

Where the *-w* and *-h* parameters specify the width and height respectfully for the window size. From this, the file *positive.vec* should exist in your current working directory.

We may now start training our model. To do so, first make a directory to append the output files to. Run the following commands:
```
mkdir my_detector
opencv_traincascade -data my_detector -vec positive.vec -bg  <neg_class>_neg.dat -numPos <N> -numNeg <M> -numStages <S> -w <width> -h <height>
```

Here, the width and height arguments are the same that you used in the last command execution. For added context and potential extensions of the above, please refer to the documentation of the OpenCV tools *opencv_createsamples* and *opencv_traincascade* [here](https://gregorkovalcik.github.io/opencv_contrib/tutorial_traincascade.html).

After running the above, you should have a working Viola Jones object detection model! The model is named *cascade.xml* and is located in the folder **my_detector**.

Now, you're able to evaluate the performance of your own detector! Try to retrieve some images and videos from Google.com or Pexels.com and run the model evaluation script to see if it performs up to standard, or whether extra data is necessary to remedy model failures.

# References
---
[YOLO object detection](https://arxiv.org/abs/1506.02640)

[Comparison of Viola-Jones performance vs. YOLO v3](https://www.researchgate.net/publication/364311838_Comparative_of_Viola-Jones_and_YOLO_v3_for_Face_Detection_in_Real_time)

[Study on the strength and limitations of the Viola-Jones algorithm](https://www.researchgate.net/publication/367584143_evaluation_study_of_face_detection_by_Viola-Jones_algorithm)