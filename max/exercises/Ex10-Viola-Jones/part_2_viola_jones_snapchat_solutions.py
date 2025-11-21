from __future__ import print_function
from typing import Tuple
import cv2
import time
import numpy as np 
from skimage.transform import rotate, rescale

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
    eyes_cascade_name = cascade_directory + "/haarcascade_eye_tree_eyeglasses.xml" #seems to be more robust to lighting and has less false detections
    
    
    if not face_cascade.load(cv2.samples.findFile(face_cascade_name)):
        print('--(!)Error loading face cascade')
        exit(0)
    if not eyes_cascade.load(cv2.samples.findFile(eyes_cascade_name)):
        print('--(!)Error loading eyes cascade')
        exit(0)

    print("Loaded cascade classifiers")
    return face_cascade, eyes_cascade

def load_png_object(file_path: str) -> Tuple[np.ndarray, Tuple[int, int, int]]:
    """
    Loads a png object as RGBA. Note, the .png should have a transparent background and be loaded with an alpha channel

    Args:
        file_path (str): relative path to the .png file 

    Returns:
        Tuple[np.ndarray, Tuple[int, int, int]]: the image array of shape (H, W, C) and a tuple (H, W, C) where C=4 for RGBA image 
    """

    im = cv2.imread(file_path,cv2.IMREAD_UNCHANGED)
    im_dims = im.shape 
    
    print(f"loaded object: {file_path} with shape: {im_dims}")
    return im, im_dims

def insert_hat_on_frame(frame_rgb, obj, coords, anchor):
    frame = frame_rgb.copy()
    
    Hf, Wf = frame.shape[:2]
    Hh, Wh = obj.shape[:2]

    hat_mask = obj[:, :, 3] > 0 #alpha "not transparent" mask
    hat_rgb  = obj[:, :, :3]

    # coords = where anchor should go in the rgb frame, i.e. face coordinate position
    target_y, target_x = coords
    anchor_y, anchor_x = anchor  # anchor inside the hat image

    # top-left corner on frame where hat-image should start
    slice_start_y = int(target_y - anchor_y)
    slice_start_x = int(target_x - anchor_x)

    # ---- STEP 1: handle negative object starts (crop top/left) ----
    crop_x_left = max(0, -slice_start_x)
    crop_y_top  = max(0, -slice_start_y)

    x1 = max(0, slice_start_x)
    y1 = max(0, slice_start_y)

    # ---- STEP 2: compute end positions before cropping bottom/right ----
    x2 = x1 + (Wh - crop_x_left)
    y2 = y1 + (Hh - crop_y_top)

    # ---- STEP 3: crop bottom/right if object is going outside frame in bottom/right ----
    crop_x_right = max(0, x2 - Wf)
    crop_y_bottom = max(0, y2 - Hf)

    # Compute final slice ranges after all cropping
    x2 = x2 - crop_x_right
    y2 = y2 - crop_y_bottom

    # Corresponding crop on hat image
    hat_crop = hat_rgb[crop_y_top:Hh - crop_y_bottom,
                       crop_x_left:Wh - crop_x_right]

    mask_crop = hat_mask[crop_y_top:Hh - crop_y_bottom,
                         crop_x_left:Wh - crop_x_right]

    # ---- STEP 4: insert ----
    im_mask = np.zeros((Hf, Wf), dtype=bool)
    full_hat = np.zeros_like(frame)

    im_mask[y1:y2, x1:x2] = mask_crop
    full_hat[y1:y2, x1:x2] = hat_crop

    frame[im_mask] = full_hat[im_mask]

    return frame

def time_smoothen_detections(arr_new: np.ndarray | float, arr_ref: np.ndarray | float, alpha: float = 0.95) -> np.ndarray | float:
    """
    Time smoothen detections. Hint: equation 4 in video change detection notes.

    Args:
        arr_new (np.ndarray or float): the input array or value to be smoothened using reference. If array shape (N,features)
        arr_ref (np.ndarray or float): the reference array or value to use for smoothing. If array shape (N,features)
        alpha (float, optional): smoothing-factor. Defaults to 0.95.

    Returns:
        np.ndarray or float: the smoothened array or value 
    """
    return (alpha*arr_ref + (1-alpha)*arr_new)
    

def rotate_object(object_im: np.ndarray, angle: float, anchor_point: tuple, allow_resize: bool = False) -> Tuple[np.ndarray,np.ndarray,int,int] | Tuple[np.ndarray,np.ndarray]:
    """
    Rotates an image using skimage.transforms.rotate around the anchor_point. 
    Finds the corresponding location of the anchor point in the rotated image frame. 
    Returns the difference in image size before and after rotation, dx and dy.  
    
    Args:
        object_im (np.ndarray): the object image to be rotated, (H, W, C)
        angle (float): rotation angle in counter-clockwise direction in degrees. 
        anchor_point (tuple): the centre in object image used rotation. For this implementation, use the bottom centre. 

    Returns:
        Tuple[np.ndarray,np.ndarray,int,int] 
    """
    h, w = object_im.shape[0], object_im.shape[1]
    object_im = rotate(object_im, angle, center=(anchor_point[1],anchor_point[0]),order=1, resize=allow_resize)
    
    #we need to know the anchor point post-translation and resizing. Easiest way is to rotate a mask containing the anchor point with the same transform used for the image 
    anchor_mask = np.zeros((h, w), dtype=np.uint8)
    anchor_mask[anchor_point] = 1
    anchor_rot = rotate(anchor_mask, angle, center=(anchor_point[1],anchor_point[0]), resize=allow_resize, order=0, preserve_range=True)
    #find the new anchor coordinate in the rotated image. This is the coordinate which needs to be mapped to the top centre of head.
    new_anchor = np.argwhere(anchor_rot == 1)[0]   # row, col in rotated space -> y, x 
    
    #calculate the difference in sizes between original image and rotated image. Used for correcting position on top of head if resize is true.
    delta_width = w-object_im.shape[1]
    delta_height = h-object_im.shape[0]
    
    return object_im, new_anchor, delta_width, delta_height 
    
    
def rescale_object(object_im: np.ndarray, face_width: int) -> np.ndarray: 
    """
    Rescales the object image to a specific width so it can match face width. 

    Args:
        object_im (np.ndarray): object image, shape (H,W,C)
        face_width (int): the detected face width

    Returns:
        np.ndarray: the rescaled object image

    """
    """
    Solution exercise 9: change the width_scale_factor such that the function adaptively fits the object onto the head
    """
    width_scale_factor = face_width/object_im.shape[1]
    out_im = rescale(object_im, width_scale_factor,channel_axis=-1, order=1)
    return out_im
    
def assign_eyes(coords: np.ndarray) -> tuple[np.ndarray,np.ndarray]:
    """
    Assign left and right eye based on location in frame

    Args:
        eye_coords (np.ndarray): eye coordinate array where each row corresponds to x,y centre coordinates, shape (N, 2) (in current implementation of main loop, N=2)

    Returns:
        tuple[np.ndarray,np.ndarray]: left eye centre coordinates, right eye centre coordinates
    """
    #Example solution: 
    right_eye_idx = np.argmax(coords[:,0]) #actually we should use the pair which has the maximal horizontal distance
    right_eye = coords[right_eye_idx,:]
    left_eye = coords[1-right_eye_idx,:]

    if left_eye[0] > right_eye[0]:
        left_eye, right_eye = right_eye, left_eye
    
    return left_eye, right_eye


if __name__=="__main__":
    ###step 1: connect to webcam
    USE_DROID_CAM = False 
    cap = connect_camera(USE_DROID_CAM)
    
    ###step 2: load classifiers 
    pretrained_dir = "pretrained_classifiers"
    face_cascade, eyes_cascade = load_cascades(pretrained_dir)
    
    # To keep track of frames per second
    start_time = time.time()
    font = cv2.FONT_HERSHEY_COMPLEX

    
    object_im, object_dims = load_png_object("image_props/hat_rescaled.png")
    object_im_trf = object_im.copy()
       
    
    #set loop parameters
    n_frames = 0
    ROTATE = True #set to true if we want to rotate the object of interest with an angle 
    SMOOTHEN = True 
    stop = False
    N_eyes = 0
    N_skipped = 0 #for counting how many frames we do not have any detections
    
    #initialize for fallback when no detections are present
    object_im_trf = None
    end_point_hat_y = None
    end_point_hat_x = None 
    anchor_point = None 

    while not stop:
        ret, frame = cap.read()
        if frame is None:
            print('--(!) No captured frame -- Break!')
            break
            
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) #convert to grayscale
        frame_gray = cv2.equalizeHist(frame_gray) 
        faces = face_cascade.detectMultiScale(frame_gray) #is a np.array (N,4) or tuple
        N_faces = len(faces)
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
                    eye_coord_arr[i,:] = np.array([eye_center[0],eye_center[1]]) #x, y in global coords
                
                left_eye, right_eye = assign_eyes(eye_coord_arr)
                """
                Solution exercise 6: find the normalized normal vector
                """
                vec = right_eye - left_eye
                ###find the normal vector - to find the translation we need  
                n_vec = np.array([-vec[1],vec[0]])
                norm = np.linalg.norm(n_vec)

                if norm == 0:  #becomes zero if the same eye is detected twice - we can't place a hat then
                    print("Warning: zero-length eye vector, skipping hat placement")
                    N_skipped += 1
                else: #normal vector has a length - continue with hat placement
                    n_vec = n_vec/norm #normalize normal-vector 
                    """
                    Solution exercise 7: calculate the end-points of the hat, i.e. where it should be placed in the image
                    """
                    end_point_hat_x = int(face_center[0] - n_vec[0]*w//2)
                    end_point_hat_y = int(face_center[1] - n_vec[1]*h//2) #subtract because image coordinate system 
                    
                    
                    cv2.circle(frame,(end_point_hat_x,end_point_hat_y),10, color=(0,255,0),thickness=3, lineType=8, shift=0) #blue circle - top of face 
                    cv2.line(frame,tuple(face_center),(end_point_hat_x,end_point_hat_y),color=(255, 0, 0 )) #green line
                    
                    """
                    Exercise 8: rescale size such that it fits the head-width 
                    """
                    object_im_trf = rescale_object(object_im, w+100) #finish the implementation of this function

                    """
                    Solution exercise 9: define the coordinates for alignment in the local hat coordinate frame
                    """
                    #anchor point calculation: bottom centre of hat 
                    anchor_point = np.array([object_im_trf.shape[0]-1,object_im_trf.shape[1]//2]) #y,x always     
                    
                    if ROTATE:      
                        """
                        Solution exercise 10: calculate the rotation angle which the hat should be rotated.
                        """ 
                        theta = np.arctan2(-vec[1],vec[0]) #is in radians
                        theta *= 180/np.pi 
                        object_im_trf, anchor_point_new, dx, dy = rotate_object(object_im_trf, theta, anchor_point, allow_resize=False) #get the corresponding anchor point after applying rotation
                        #TODO: somehow anchor_point_new needs to be part of the transform when allow_resize=True, but it seems okay-ish
                        anchor_point[0] = anchor_point[0] - dy 
                        anchor_point[1] = anchor_point[1] - dx 
                    frame = insert_hat_on_frame(frame, object_im_trf, (end_point_hat_y,end_point_hat_x), anchor_point) 
                    N_skipped = 0
            else: #case: no valid detection, fall-back to references 
                N_skipped += 1
                if object_im_trf is not None and end_point_hat_y is not None and end_point_hat_x is not None and anchor_point is not None:
                    frame = insert_hat_on_frame(frame, object_im_trf, (end_point_hat_y,end_point_hat_x), anchor_point) 
        
            # Keep track of frames-per-second (FPS) and also show skipped frames information
            n_frames = n_frames + 1
            elapsed_time = time.time() - start_time
            fps = int(n_frames / elapsed_time)
            # Put the information on the image frame: FPS, number of faces, number of eyes 
            str_out = f"fps: {fps}, N_skip: {N_skipped}"
            cv2.putText(frame, str_out, (0, 30), font, 1, 255, 1)
            cv2.imshow('Capture - Face detection', frame)
            
            if cv2.waitKey(1) == ord('q'):
                stop = True


