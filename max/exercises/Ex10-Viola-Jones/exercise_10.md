
# Introduction
In OpenCV, the task of Viola-Jones-style object detection is referred to as cascade classification. We will use the two terms interchangeably, although there are a few differences as compared to the original Viola-Jones paper. Disregarding details, from an intuitive standpoint, the two approaches are almost the same.


# Exercises outline
- In part 1 you will gain familiarity with some pretrained Viola-Jones type detectors provided by OpenCV and learn how to use them
- In part 2 you will build a Snapchat-inspired object-tracking filter for pasting an object onto a face
- In part 3 you will work on intuitively interpreting the building blocks of Viola-Jones, namely the Haar-features, and relate the scale-parameter to the detection quality. You are also given the opportunity to optionally train your own classifier.

# Learning goals (not finished, still to be checked)

- Be able to use opencv pretrained cv::CascadeClassifier()
- Count the number of detections from a pretrained cv::CascadeClassifier
- Compute outputs of haar features using an integral image
- Interpret how simple Haar-features relate to common properties of the class to be detected
- Identify failure modes of Viola-Jones with respect to simple image conditions such as lighting, object transformations, distance and resolution
- Relate the scale of a Viola-Jones object detector to a specific task, considering object sizes


# Part one: warming up and gaining familiarity 

Before solving these exercises, you will need to download pretrained cascade classifiers from https://raw.githubusercontent.com/opencv/opencv/3.4/data/haarcascades/. We will need:
- haarcascade_frontalface_default.xml 
- haarcascade_eye.xml
- haarcascade_eye_tree_eyeglasses.xml 

In in the following exercises. 
Save them somewhere where you can easily define the path for loading, e.g. in a folder called pretrained_classifiers



## Exercise 1: using a pretrained classifier on web-cam feed 
Start out by opening the file part_one_viola_jones.py. 
Change the variable pretrained_dir such that it points to your pretrained classifiers. Gain a quick overview of the script - which functions do what? Where is the detection actually happening? Try and run the script and moving a bit around in front of the webcam, rotating your head, looking to the sides, and changing your distance to the screen. When does the detection work well? 

## Exercise 2: Counting the number of detections
Implement functionality to count the number of faces and eyes detected. This should be based on the output of the detectMultiScale function calls, i.e. the variables faces and eyes. Print the number of eyes and faces detected into the text which already measures the frames per second. Ensure that it works correctly when multiple faces with eyes are detected.    

*Hint: In order to figure out what the function returns, try to write print(faces) when a face is detected vs when you put a finger in front of the webcam. Is the return type consistent in the two cases?*
 
*Hint2: len() both works both for numpy arrays and python lists and tuples.*

## Exercise 3: Decreasing false positive and false negative detections
You may notice a number of false positive detections, especially for the eye detection. Likewise, if you wear glasses, you may experience some false negatives. 
Let's try to improve upon that. There are two quick fixes we will try out:

1. Check out the detectMultiScale() documentation here: https://docs.opencv.org/3.4/d1/de5/classcv_1_1CascadeClassifier.html. We can set restrict detections to be larger than a certain size. Try to tune the size such that nostrils are no longer detected. You can also set a maxSize, if many detections are larger than what you expect eyes would be. Both should ideally be set adaptively based on the height and width of the face detected. 
2. Using another trained detection model. We could e.g. try the haarcascade_eye_tree_eyeglasses.xml model instead, which is trained to be more robust. Change the function load_cascades so that you use the eyeglasses-model instead and see if this improves robustness. Try comparing the quality when the minSize and maxSize arguments are set vs. not set. 

You could of course also implement these changes into the face detection, the quality of which also is based on lighting conditions, etc. 

# Part two: Snapchat-like object filters 
We've already seen that we can detect eyes and faces somewhat robustly. Now, we will use this knowledge to make a Snapchat-like filter, where an object is placed on an image (the webcam feed) on an anchor-point (e.g. the top of the head). The goal is, that the object should track the anchor-point on the head through successive frames. 


In order to achieve this, we will need to be able to load an image with either a transparent background (RGBA-image), or alternatively load an image which we can make a mask for which could achieve the same end.

## Exercise 4: loading an RGBA image
Open the file part_2_viola_jones_snapchat.py.

Finish the function load_png_object and use it to load the hat object which has the path image_props/hat_rescaled.png. pass the flag cv2.IMREAD_UNCHANGED, which forces openCV to load the image using an alpha-channel. Return the image and the image shape. Test that the shape is as you expect. How can you interpret the alpha-channel?

## Exercise 5: Eye-centre landmark registration 
For the approach we will use for visualizing the hat on top of the head, we will use a simplified landmark-based approach. In order to place the hat correctly, we will need to infer three landmarks: 1. the centre coordinate of the top of the detected face, 2. the centre coordinate of the left eye detection and 3. the centre coordinate of the right eye detection. 
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

Your task is to finish the function assign_eyes which should output the left and right eye centre coordinates as np.arrays 

## Exercise 6: Constructing a normal vector for finding the top of the face 
Now, based on the left and right eye centre coordinates, we want to identify the top of the head. This can be examplewise be done by finding a vector which always points upwards in the face coordinate system. This can be achieved by first finding the vector which goes from the left eye to the right eye, and followingly finding its normalized normal vector. 


Your task is to: 
1. Calculate the vector going from left eye to right eye
2. Find the corresponding normal vector 
3. Calculate the normalizing factor such that it is a unit-norm normal vector
   
The normalizing factor can be found by using np.linalg.norm(n_vec).

Note: in some cases, the CascadeClassifier registers multiple eye instances in the same eye. In this case, the norm will be zero, and normalization will yield an error. As a result, we will in this case not proceed further with calculating the normalized normal vector.

## Exercise 7: Identifying the coordinate at the top-middle of the face
If we from the centre coordinate of the face go along the normal vector along the length h//2, where h is the height of the face, we should reach the top coordinate of the face. This will yield our third landmark. 

Calculate the end-point on the face. Ensure that the output is an integer, and if not, round it to an integer. 
You can debug your solution with: 
```python
cv2.circle(frame,(end_point_hat_x,end_point_hat_y),10, color=(0,255,0),thickness=3, lineType=8, shift=0) #blue circle - top of face 
``` 

which inserts a circle at the coordinate you've found. 


## Exercise 8: Transformations!
A snapchat filter resizes the objects such that they somewhat fit with the width of the head, so we will do the same - currently, it is way too small! Finish the function rescale_object such that the hat somewhat fits on the face. Use skimage.transforms.rescale for this. 


## Exercise 9: Place the hat on the coordinate
In order to place the hat, we need to identify which coordinate in the hat coordinate frame we wish to align with the found end_point_hat_x and end_point_hat_y. In other words, we need to find the necessary translation between the hat coordinate frame (local) and the web-cam image coordinate frame (global). We save these coordinates in format y,x in the anchor_point variable. 
Try defining a suitable alignment point and visualizing the results by running the script.
*hint: A good coordinate could be the bottom centre of the hat image* 


Note that the transformation is carried out in the insert_hat_on_frame function, where the variables slice_start_y and slice_start_x:
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


## Exercise 10: Rotating for realism
Based on our normal-vector, we can even calculate the corresponding angle the hat should be rotated to follow the face rotation. The angle of rotation is measured in counter-clockwise direction compared to the horizontal axis, and as such can be calculated using the normal-vector. 

Set the boolean ROTATE flag as True and implement the calculation of the rotation angle. Store the result in theta, and ensure it is in degrees. 
The rotate_object function rotates the object using the anchor_point as rotation centre with the skimage.transform.rotate function we've worked with before. Use the center bottom point of the object image as the centre coordinate, i.e.: 
```python
anchor_point = np.array([object_im_trf.shape[0]-1,object_im_trf.shape[1]//2])
```

The function outputs the position of the rotation-centre post transformation aswell. 

*note: for larger rotation angles the object will become cropped. If you set the flag allow_resize=True, the rotated object image is resized such that no information is lost. This adds a layer of complexity, as we in this case need to correct for the resizing of the object as well. This is handled by the variables dy and dx*


## Exercise 11: Speeding things up (Optional)
There are quite a few inefficiencies in the code. The largest overhead is in the image transformations, so reducing the number of times they are called or making them more efficient would improve real-time performance. 
Here are a few of the more obvious suggestions for increasing performance: 
- The interpolation order for rotations and rescalings could in some cases be set to 0 with no large effect on the output. This is cheaper to compute 
- Rotating an object can be fairly expensive, especially if it is a large image. Similarly resizing is not always necessary. We could spare the transformations if we chose to only resize when the face width is significantly different to last iteration. Similarly for the rotation angle. Else we could simply use the transformed image from the last iteration. 
- While good, skimage.transform.rotate and skimage.transform.rescale are significantly slower than opencv alternatives. Try to implement some of the functionality using opencv instead.
- As we learned in the week with geometric transformations, transformations can be combined. As a result, we could combine the translation, scaling and rotation into a single rigid transformation. Of course, you would need to keep track of where the alignment centre ends up in the transformed object image
- The rotation of the anchor-point mask is currently done numerically, thus making one unecessary rotation per iteration. Could you find the coordinate analytically instead?

You can try any of these optimizations (there exist more than these), and see if you can measure the impact on the fps displayed. 

## Exercise 12: Increasing robustness (Optional)
In some lighting conditions, the hat placement is still very unrobust, and eye-detection may still be faulty. Try to tune the max and min-size of face and eye detections. In addition, you could build in "memory" of earlier rotation angles theta and the end_point_hat_y and end_point_hat_x positions. Hint: consider equation 4 from the video-change-detection note. Finish the time_smoothen_detections function and add smoothing support. Does your result improve? 
Consider the reason why we calculate the smoothing only on the end-points and angle, and not on the intermediary quantities. 



#STILL TODOS: 
- part_2: when ROTATE and allow_resize=True, the transformation still is not completely correct. Would be nice for the students to have a correct transform. 
  - It is mainly an issue when rotation angle is negative. Det er ikke helt "så hatten passer"!
- part_2: implement smoothing in solutions. 
- part_2: add more comments / documentation on insert_hat_on_frame function
- part_2: add more objects for students to try
- part_2: add sources for data files in .txt 









    