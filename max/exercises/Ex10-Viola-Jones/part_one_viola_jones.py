from __future__ import print_function
from typing import Tuple
import time
import cv2

"""
An adapted version of https://docs.opencv.org/3.4/db/d28/tutorial_cascade_classifier.html
"""

def connect_camera(use_droid_cam: bool = False) -> cv2.VideoCapture:
    """
    Attempts to connect to the webcam. 
    If this fails, prints an error message and exits the complete script. 

    Args:
        use_droid_cam (bool, optional): whether to use the android camera, as done in video change detection exercises, ex2b. Defaults to False.

    Returns:
        cv2.VideoCapture: the opencv camera object which we can pull frames from
    """
    print("Opening connection to camera")
    url = 0
    use_droid_cam = False
    if use_droid_cam:
        url = "http://192.168.1.120:4747/video"
    cap = cv2.VideoCapture(url)
    if not cap.isOpened():
        print("Cannot open camera")
        exit()

    ret, _ = cap.read()
    if not ret:
        print("---(!)Error reading frame from webcam")
        exit()
    print("Successfully connected to camera capturing device")
    return cap 

def load_cascades(cascade_directory: str) -> Tuple[cv2.CascadeClassifier, cv2.CascadeClassifier]:
    """
    A function to load a the pretrained opencv cascade detectors (Viola-Jones-type)
    Remember to download the classifiers beforehand (https://raw.githubusercontent.com/opencv/opencv/3.4/data/haarcascades/)

    Args:
        cascade_directory (str): the directory the classifier .xml files are in

    Returns:
        tuple[cv2.CascadeClassifier]: a tuple of the two classifiers
    
    """
    face_cascade = cv2.CascadeClassifier() #initialize a classifier instance
    eyes_cascade = cv2.CascadeClassifier()
    
    face_cascade_name = cascade_directory + "/haarcascade_frontalface_default.xml"
    eyes_cascade_name = cascade_directory + "/haarcascade_eye.xml" #specify which classifiers to load
    
    
    if not face_cascade.load(cv2.samples.findFile(face_cascade_name)):
        print('--(!)Error loading face cascade')
        exit(0)
    if not eyes_cascade.load(cv2.samples.findFile(eyes_cascade_name)):
        print('--(!)Error loading eyes cascade')
        exit(0)

    print("Loaded cascade classifiers")
    return face_cascade, eyes_cascade


if __name__=="__main__":
    ###step 1: connect to webcam
    USE_DROID_CAM = False 
    cap = connect_camera(USE_DROID_CAM)
    
    ###step 2: load classifiers, remember to download them beforehand 
    pretrained_dir = "pretrained_classifiers"
    face_cascade, eyes_cascade = load_cascades(pretrained_dir)
    
    # To keep track of frames per second
    start_time = time.time()
    font = cv2.FONT_HERSHEY_COMPLEX
    n_frames = 0

    #make the loop
    stop = False

    while not stop:
        ret, frame = cap.read() #read rgb frame
        if frame is None:
            print('--(!) No captured frame -- Break!')
            break
    
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) #convert to grayscale
        frame_gray = cv2.equalizeHist(frame_gray) #see pp 42-44 in MIA book. Necessary because the pre-trained classifiers were trained on this
        

        faces = face_cascade.detectMultiScale(frame_gray) #face detection
        for (x,y,w,h) in faces:
            center = (x + w//2, y + h//2)
            frame = cv2.ellipse(frame, center, (w//2, h//2), 0, 0, 360, (255, 0, 255), 4)
            faceROI = frame_gray[y:y+h,x:x+w]
            eyes = eyes_cascade.detectMultiScale(faceROI) #constrain eye-detection to be within the face region
            for (x2,y2,w2,h2) in eyes:
                eye_center = (x + x2 + w2//2, y + y2 + h2//2)
                radius = int(round((w2 + h2)*0.25))
                frame = cv2.circle(frame, eye_center, radius, (255, 0, 0 ), 4)
        
        # Keep track of frames-per-second (FPS)
        n_frames = n_frames + 1
        elapsed_time = time.time() - start_time
        fps = int(n_frames / elapsed_time)

        """
        exercise 2) count the number of detected faces and eyes
        """
        N_faces = -1
        N_eyes = -1

        # Put the information on the image frame: FPS, number of faces, number of eyes 
        str_out = f"fps: {fps}, N_f: {N_faces}, N_e: {N_eyes}"
        cv2.putText(frame, str_out, (100, 100), font, 1, 255, 1)
        cv2.imshow('Capture - Face detection', frame)

        if cv2.waitKey(1) == ord('q'):
            stop = True


