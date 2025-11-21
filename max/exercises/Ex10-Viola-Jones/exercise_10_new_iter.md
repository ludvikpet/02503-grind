
# Exercise 10 - Real Time Object Detection Using Viola Jones Method

## Introduction
---
In this exercise, we take a look at the use case of the widely deployed Viola-Jones algorithm - a method that enables real-time face detection. Object detection is a core problem within computer vision, and with its introduction in 2001, Viola Jones became the first face detection algorithm used in real-time. Even for its ancient beginnings, it is still applicable today due to its relatively high accuracy performance in conjunction with its low compute requirement, which is still competitive with much later frameworks such as YOLO v3. The algorithm has later been integrated within large applications such as *Snapchat*, which use the framework within their popular *Snapchat Lenses* widget. Later on, we'll explore how face detection can be adopted for such uses. Similarly we'll show, that the algorithm can be adapted to detect other object classes.

![[viola_jones_face_detection.png]]
Figure retrieved from the [Viola Jones paper](https://ieeexplore.ieee.org/document/990517).

In OpenCV, the task of Viola-Jones-style object detection is referred to as cascade classification. We will use the two terms interchangeably, although there are a few differences as compared to the original Viola-Jones paper. Disregarding details, from an intuitive standpoint, the two approaches are almost the same.


## Exercises outline
---
- In part 1 you will gain familiarity with some pretrained Viola-Jones type detectors provided by OpenCV and learn how to use them
- In part 2 you will build a Snapchat-inspired object-tracking filter for pasting an object onto a face
- In part 3 you will work on intuitively interpreting the building blocks of Viola-Jones, namely the Haar-features, and relate the scale-parameter to the detection quality.
- In part 4 you will extract the value of a chosen Haar feature, gaining an understanding of the underlying mechanism of Viola Jones feature extraction.
- In part 5 you'll be given the opportunity to **optionally** train your own classifier.

## Learning goals (not finished, still to be checked)
---
- Be able to use opencv pretrained cv2::CascadeClassifier()
- Count the number of detections from a pretrained cv2::CascadeClassifier
- Compute outputs of haar features using an integral image
- Interpret how simple Haar-features relate to common properties of the class to be detected
- Identify failure modes of Viola-Jones with respect to simple image conditions such as lighting, object transformations, distance and resolution
- Relate the scale of a Viola-Jones object detector to a specific task, considering object sizes

## Part 1: warming up and gaining familiarity 
---
Before solving these exercises, you will need to download pretrained cascade classifiers from this [link](https://raw.githubusercontent.com/opencv/opencv/3.4/data/haarcascades/). We will need:
- haarcascade_frontalface_default.xml 
- haarcascade_eye.xml
- haarcascade_eye_tree_eyeglasses.xml 

In the following exercises. 
Save them somewhere where you can easily define the path for loading, e.g. in a folder called pretrained_classifiers

### Exercise 1: using a pretrained classifier on web-cam feed 
Start out by opening the file *part_one_viola_jones.py*. Change the variable *pretrained_dir* such that it points to your pretrained classifiers and gain a quick overview of the script.

**Question 1**: *Which functions do what? Where is the detection actually happening?*

Now run the script and try moving a bit around in front of the webcam, e.g. by rotating your head, looking to the sides, and changing your distance to the screen.

**Question 2**: *When does the detection work well, and can you tell why the model has deficiencies?* 

### Exercise 2: Counting the number of detections
Implement functionality to count the number of faces and eyes detected. This should be based on the output of the detectMultiScale function calls, i.e. the variables faces and eyes. Print the number of eyes and faces detected into the text which already measures the frames per second. Ensure that it works correctly when multiple faces with eyes are detected.    

*Hint: In order to figure out what the function returns, try to write print(faces) when a face is detected vs when you put a finger in front of the webcam. Is the return type consistent in the two cases?*
 
*Hint2: len() both works both for numpy arrays and python lists and tuples.*

### Exercise 3: Decreasing false positive and false negative detections
You may notice a number of false positive detections, especially for the eye detection. Likewise, if you wear glasses, you may experience some false negatives. 
Let's try to improve upon that. There are two quick fixes we will try out:

1. Check out the *detectMultiScale()* documentation [here](https://docs.opencv.org/3.4/d1/de5/classcv_1_1CascadeClassifier.html). We can restrict detections to be larger than a certain size. Try to tune the size such that nostrils are no longer detected. You can also set a maxSize, if many detections are larger than what you expect eyes would be. Both should ideally be set adaptively based on the height and width of the face detected. 
2. Using another trained detection model. We could e.g. try the *haarcascade_eye_tree_eyeglasses.xml* model instead, which is trained to be more robust. Change the function *load_cascades* so that you use the eyeglasses-model instead and see if this improves robustness. Try comparing the quality when the minSize and maxSize arguments are set vs. not set. 

You could of course also implement these changes into the face detection, the quality of which also is based on lighting conditions, etc. 

## Part 2: Snapchat-like object filters 
---
We've already seen that we can detect eyes and faces somewhat robustly. Now, we will use this knowledge to make a Snapchat-like filter, where an object is placed on an image (the webcam feed) on an anchor-point (e.g. the top of the head). The goal is, that the object should track the anchor-point on the head through successive frames. 


In order to achieve this, we will need to be able to load an image with either a transparent background (RGBA-image), or alternatively load an image which we can make a mask for which could achieve the same end.

### Exercise 4: loading an RGBA image
Open the file *part_2_viola_jones_snapchat.py*. Finish the function *load_png_object* and use it to load the hat object which has the path *image_props/hat_rescaled.png. pass the flag cv2.IMREAD_UNCHANGED, which forces openCV to load the image using an alpha-channel. Return the image and the image shape. Test that the shape is as you expect.

**Question 3**: *How can you interpret the alpha-channel?*

### Exercise 5: Eye-centre landmark registration 
For the approach we will use for visualizing the hat on top of the head, we will use a simplified landmark-based approach. In order to place the hat correctly, we will need to infer three landmarks: **1)** the centre coordinate of the top of the detected face, **2)** the centre coordinate of the left eye detection and **3)** the centre coordinate of the right eye detection. 
For each detected face and pair of eyes, we will need to save these coordinates.

All quantities we need to infer the landmarks are defined in the following loop:

```python 
for i, (x,y,w,h) in enumerate(faces):
            center = (x + w//2, y + h//2)
            face_center = np.array([center[0],center[1]])
            frame = cv2.ellipse(frame, center, (w//2, h//2), 0, 0, 360, (255, 0, 255), 4)
            faceROI = frame_gray[y:y+h,x:x+w]
            #detect eyes
            eyes = eyes_cascade.detectMultiScale(faceROI)
            N_eyes = len(eyes)
            eye_center_holder = []
            if N_eyes==2: #only go further if we have detected two eyes.. 
                eye_coord_arr = np.zeros((2,2))
                for i, (x2,y2,w2,h2) in enumerate(eyes):
                    eye_center = (x + x2 + w2//2, y + y2 + h2//2)
                    radius = int(round((w2 + h2)*0.25))
                    frame = cv2.circle(frame, eye_center, radius, (255, 0, 0 ), 4)
                    eye_coord_arr[i,:] = np.array([eye_center[0],eye_center[1]]) #x, y in frame coordinates
```

First, we will start by registering the left and right eye centre coordinates.
If we work from a simplified assumption that the right eye will be located to the right of the left eye, we can identify the centre coordinates based on the contents of  eye_coord_arr. 

Your task is to finish the function *assign_eyes* which should output the left and right eye centre coordinates as np.arrays.

### Exercise 6: Constructing a normal vector for finding the top of the face 
Now, based on the left and right eye centre coordinates, we want to identify the top of the head. This can examplewise be done by finding a vector which always points upwards in the face coordinate system. This can be achieved by first finding the vector which goes from the left eye to the right eye, and followingly finding its normalized normal vector. 


Your task is to: 
1. Calculate the vector going from left eye to right eye
2. Find the corresponding normal vector 
3. Calculate the normalizing factor such that it is a unit-norm normal vector
   
The normalizing factor can be found by using np.linalg.norm(n_vec).

**Note**: in some cases, the CascadeClassifier registers multiple eye instances in the same eye. In this case, the norm will be zero, and normalization will yield an error. As a result, we will in this case not proceed further with calculating the normalized normal vector.

### Exercise 7: Identifying the coordinate at the top-middle of the face
If we from the centre coordinate of the face go along the normal vector along the length $\frac{h}{2}$, where $h$ is the height of the face, we should reach the top coordinate of the face. This will yield our third landmark. 

Calculate the end-point on the face. Ensure that the output is an integer, and if not, round it to an integer. 
You can debug your solution with: 
```python
cv2.circle(frame,(end_point_hat_x,end_point_hat_y),10, color=(0,255,0),thickness=3, lineType=8, shift=0) #blue circle - top of face 
``` 

which inserts a circle at the coordinate you've found. 

### Exercise 8: Transformations!
A snapchat filter resizes the objects such that they somewhat fit with the width of the head, so we will do the same - currently, it is way too small! Finish the function *rescale_object* such that the hat somewhat fits on the face. Use *skimage.transforms.rescale* for this. 

### Exercise 9: Place the hat on the coordinate
In order to place the hat, we need to identify which coordinate in the hat coordinate frame we wish to align with the found end_point_hat_x and end_point_hat_y. In other words, we need to find the necessary translation between the hat coordinate frame (local) and the web-cam image coordinate frame (global). We save these coordinates in format y,x in the anchor_point variable. 
Try defining a suitable alignment point and visualizing the results by running the script.
*hint: A good coordinate could be the bottom centre of the hat image* 


**Note**: The transformation is carried out in the *insert_hat_on_frame* function, where the variables *slice_start_y* and *slice_start_x*:
```python
def insert_hat_on_frame(frame_rgb, obj, coords, anchor):
    # coords = where anchor should go in the rgb frame, i.e. face coordinate position
    target_y, target_x = coords
    anchor_y, anchor_x = anchor  # anchor inside the hat image

    # top-left corner on frame where hat-image should start
    slice_start_y = int(target_y - anchor_y) 
    slice_start_x = int(target_x - anchor_x)
```
define at which global coordinate the hat image (0,0) should be located. 


### Exercise 10: Rotating for realism
Based on our normal-vector, we can even calculate the corresponding angle the hat should be rotated to follow the face rotation. The angle of rotation is measured in counter-clockwise direction compared to the horizontal axis, and as such can be calculated using the normal-vector. 

Set the boolean *ROTATE* flag as True and implement the calculation of the rotation angle. Store the result in *theta*, and ensure it is in degrees. 
The *rotate_object* function rotates the object using the *anchor_point* as rotation centre with the *skimage.transform.rotate* function we've worked with before. Use the center bottom point of the object image as the centre coordinate, i.e.: 
```python
anchor_point = np.array([object_im_trf.shape[0]-1,object_im_trf.shape[1]//2])
```

The function outputs the position of the rotation-centre post transformation aswell. 

**Note**: For larger rotation angles the object will become cropped. If you set the flag allow_resize=True, the rotated object image is resized such that no information is lost. This adds a layer of complexity, as we in this case need to correct for the resizing of the object as well. This is handled by the variables dy and dx


### Exercise 11: Speeding things up (Optional)
There are quite a few inefficiencies in the code. The largest overhead is in the image transformations, so reducing the number of times they are called or making them more efficient would improve real-time performance. 
Here are a few of the more obvious suggestions for increasing performance: 
- The interpolation order for rotations and rescalings could in some cases be set to 0 with no large effect on the output. This is cheaper to compute 
- Rotating an object can be fairly expensive, especially if it is a large image. Similarly resizing is not always necessary. We could spare the transformations if we chose to only resize when the face width is significantly different to last iteration. Similarly for the rotation angle. Else we could simply use the transformed image from the last iteration. 
- While good, skimage.transform.rotate and skimage.transform.rescale are significantly slower than opencv alternatives. Try to implement some of the functionality using opencv instead.
- As we learned in the week with geometric transformations, transformations can be combined. As a result, we could combine the translation, scaling and rotation into a single rigid transformation. Of course, you would need to keep track of where the alignment centre ends up in the transformed object image
- The rotation of the anchor-point mask is currently done numerically, thus making one unecessary rotation per iteration. Could you find the coordinate analytically instead?

You can try any of these optimizations (there exist more than these), and see if you can measure the impact on the fps displayed. 

### Exercise 12: Increasing robustness (Optional)
In some lighting conditions, the hat placement is still very unrobust, and eye-detection may still be faulty. Try to tune the max and min-size of face and eye detections. In addition, you could build in "memory" of earlier rotation angles theta and the end_point_hat_y and end_point_hat_x positions. Hint: consider equation 4 from the video-change-detection note. Finish the time_smoothen_detections function and add smoothing support. Does your result improve? 
Consider the reason why we calculate the smoothing only on the end-points and angle, and not on the intermediary quantities. 



#STILL TODOS: 
- part_2: when ROTATE and allow_resize=True, the transformation still is not completely correct. Would be nice for the students to have a correct transform. 
  - It is mainly an issue when rotation angle is negative. Det er ikke helt "så hatten passer"!
- part_2: implement smoothing in solutions. 
- part_2: add more comments / documentation on insert_hat_on_frame function
- part_2: add more objects for students to try
- part_2: add sources for data files in .txt 

## Part 3: Cascade Classifier used for object detection
---
In the following exercises, we'll investigate the performance of the Viola-Jones algorithm when trained to detect non-faces. To do this, we've trained a stop-sign object detector, which we'll use in the following exercises. Later on, you'll be able to try such a model yourselves (optional). We'll show, that the framework doesn't necessarily need to be used for faces, but that it is instead a general object detection tool which can be trained to solve any object detection problem.

We'll start by employing a pretrained stop-sign object detector for image capture. To do this, we'll use the script **VJ_performance_eval.py**. You can find all the scripts and data we'll need [here](linkidinky.com). Download the folder and *cd* into it.
### Exercise 13: Testing stop-sign model performance
We'll first investigate how the model performs on a simple image of a stop sign. in the **data** folder, the image *stop_sign.jpg* can be found. First, try to visualize the image. We may evaluate the model performance using the beforementioned *VJ_performance_eval.py* script and running the following command in your terminal:

```
python VJ_performance_eval.py --model stop_sign_detector.xml --input stop_sign.jpg --type image --w 24 --h 24
```

**Question 4**: *Does the model perform as expected?*

**Question 5**: *Inspect the script contents and check how the model configures its bounding box shape. Now try to print the values. Specify the top left and bottom right points of the bounding box frame.*


### Exercise 14: Visual inspection of Haar features
In order to understand the underlying decision logic that the model makes during execution, we'll take a closer look at the computed Haar features that the model accepts during evaluation. Refer to the directory **haar_features** to get insights into what the model has learned to put attention towards at different stages when given the sample image as input.

**Question 6**: *Looking at some of the computed haar features, can you tell what the model attends to at the different classification stages?*

### Exercise 15: Video stream object detection
Now we'd like to look at the performance of the stop sign detector when tasked with a moving image, to see whether it is robust to real-time image changes, such as object scaling and pose adjustments. Again run the file *VJ_performance_eval.py*, now using the videos **van_video.mp4** and **scale_diff.mp4** as image capture input. Remember to change the type of input to video format.

**Question 7**: *How does the model perform? Can you find any weaknesses that it possesses?*

During detection, the model computes features based on a series of scaled image sizes, forming a pyramid of images. OpenCV allows us to modify this pyramid through their detection framework.

**Question 8**: *Examine the script file and try to modify the scale_factor parameter of the detection function. Explain the effects of increasing and decreasing the scale factor.*

## Part 4: Haar feature diagnostics
---
We'd now like to take a deeper look at what the model actually sees when computing a Haar feature. Again, we'll refer to the **haar_features** directory - choose a cascade classifier/Haar feature that you find interesting from one of the stages within the directory. You'll use this image throughout this section.

### Exercise 16: Loading and resizing image (Probably should be removed)
Given the chosen Haar feature, slice the image by removing the other stage features, and resize it to 24x24. Now visualize the resized image along with the image *annotation_img.jpg*, which is the original reference image.

Visualization of the cascade classifiers/Haar regions was made using OpenCV2's CLI tool *opencv_visualisation*. This tool has some limitations, namely, the input image has to be of shape $(w_{window}, h_{window})$, yet the output image is upscaled x10. Upscaling adds some artifacts to the image (e.g. a decrease in contrast), but above all, it necessitating a resizing for our task, as we need to refer to a reference image. Our attention will therefore be on the resized- and reference image.

For later use, remember to grayscale the reference image.

**Note:** *If you want to visualize cascade classifiers on your own image annotation, look at the documentation of the beforementioned function [here](https://docs.opencv.org/3.4/dc/d88/tutorial_traincascade.html).*

### Exercise 17: Identify ROI pixels
Having resized our image, we now want to slice our *reference* image such, that we only tend to the pixels that are contained within the chosen haar feature. Find these pixels on your resized image and slice the reference image accordingly.

*Hint: Histograms could be a valuable asset.*

### Exercise 18: Compute the integral image of image slice
Having found the ROI, compute the integral image over the image slice. Visualize the integral image along with the original image slice.

*Hint: The functions numpy.cumsum or cv2.integral may be of use.*

**Note**: *It is important that the boundary of your integral image goes beyond the ROI by at least 1 pixel in all directions. cv2.integral fixes this for you by zero-padding the boundary, but keep this in mind when using numpy.cumsum.*

### Exercise 19: Compute the Haar feature
Now, compute your Haar feature. To do so, extract the haar corners of each Haar region and compute their individual Haar sum. Finally, compute:

$$H = [\text{Sum of regions with white pixels}] - [\text{Sum of regions with black pixels}]$$

To retrieve the Haar feature. The following figure might become of use:

![[haar_computation.png]]
Figure copied from [Wikipedia](https://en.wikipedia.org/wiki/Summed-area_table).

## Part 5: Training your own object detector (Optional)
---
The stop-sign model was trained using data from Caltech-101, consisting of 64 training images with bounding box annotations. We'll now try to train our own object detector from the same dataset! The dataset contains 101 annotated object categories each with separate subset sizes, ranging from 40-800 images. You can download the dataset from [here](https://data.caltech.edu/records/mzrjq-6wc02). In order for everything to work as expected, you need to place the root of the Caltech-101 dataset into the **data** folder.

In order to perform this exercise, we highly recommend using the Anaconda Python distribution. Training an object detector using tools provided by OpenCV requires an older version of OpenCV, which can be handled much easier using Anaconda.

### Exercise 20: Setup new virtual environment (Optional)
We first need to setup a new virtual environment, in order to run the tools we need, which have been disabled for newer versions of OpenCV (>3.4) . To start with, if you're currently in a virtual environment, deactivate by running:

```bash
conda deactivate
```

Now we create a virtual environment with the necessary installation of OpenCV and activate it. This can be done as follows:

```bash
conda create -n obj_train python 'opencv>=3,<4'
conda activate obj_train
```

### Exercise 21: Preprocess data (Optional)
In order for our data to fit to the semantic notation needed for the OpenCV training tools, we first need to perform some pre-processing on our annotations. To do so, you need to choose **1)** a *positive* class, denoting the class you want your model to identify; and **2)** a *negative* class, denoting a mock class intended to act as a model *regularizer*, meaning that we try to nudge our model to learn relevant features instead of erroneous features. In the case of the stop-sign model, the chosen positive class was *stop_sign* and the negative class was *car_side*.

Run the following command when at the root of the **process_dataset.py** script:

```bash
python process_dataset.py --pos <pos_class> --neg <neg_class>
```

With *pos_class* and *neg_class* being the explicit names of the categories in the Caltech-101 dataset.

Having ran this script, the description files *pos_class_pos.dat* and *pos_class_neg.dat* should be contained in the Caltech-101 folder along with their gray-scaled datasets in the folders **pos_class_gray** and **neg_class_gray**.

### Exercise 22: Train object detector (Optional)
Now we'll commence the training portion of the exercise! To start with, *cd* into the Caltech-101 folder, as the following OpenCV tools only work from the folder where the data resides. We need to create a *positive vector file*, which provides the path from the positive images to the positive description file. This can be done as follows:

```bash
opencv_createsamples -info <pos_class>_pos.dat -vec positive.vec -w <width> -h <height>
```

Where the *-w* and *-h* parameters specify the width and height respectfully for the window size. From this, the file *positive.vec* should exist in your current working directory.

We may now start training our model. To do so, first make a directory to append the output files to. Run the following commands:
```bash
mkdir my_detector
opencv_traincascade -data my_detector -vec positive.vec -bg  <neg_class>_neg.dat -numPos <N> -numNeg <M> -numStages <S> -w <width> -h <height>
```

Here, the width and height arguments are the same that you used in the last command execution. For added context and potential extensions of the above, please refer to the documentation of the OpenCV tools *opencv_createsamples* and *opencv_traincascade* [here](https://gregorkovalcik.github.io/opencv_contrib/tutorial_traincascade.html).

After running the above, you should have a working Viola Jones object detection model! The model is named *cascade.xml* and is located in the folder **my_detector**.

Now, you're able to evaluate the performance of your own detector! Try to retrieve some images and videos from e.g. [Google.com](https://www.google.com/) or [Pexels.com](https://www.pexels.com/) and run the model evaluation script to see if it performs up to standard, or whether extra data is necessary to remedy model failures.

# References
---
[YOLO object detection](https://arxiv.org/abs/1506.02640)

[Comparison of Viola-Jones performance vs. YOLO v3](https://www.researchgate.net/publication/364311838_Comparative_of_Viola-Jones_and_YOLO_v3_for_Face_Detection_in_Real_time)

[Study on the strength and limitations of the Viola-Jones algorithm](https://www.researchgate.net/publication/367584143_evaluation_study_of_face_detection_by_Viola-Jones_algorithm)







    