from __future__ import print_function
import cv2
import argparse
import time
import numpy as np 
from skimage.transform import rotate, rescale


def connect_camera(use_droid_cam: bool = False) -> cv2.VideoCapture:
    print("Opening connection to camera")
    url = 0
    use_droid_cam = False
    if use_droid_cam:
        url = "http://192.168.1.120:4747/video"
    cap = cv2.VideoCapture(url)
    # cap = cv22.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open camera")
        exit()

    ret, frame = cap.read()
    # if frame is read correctly ret is True
    if not ret:
        print("---(!)Error reading frame from webcam")
        exit()
    print("Successfully connected to camera capturing device")
    return cap 

def load_cascades(cascade_directory: str) -> tuple[cv2.CascadeClassifier]:
    face_cascade = cv2.CascadeClassifier()
    face_cascade_name = pretrained_dir + "/haarcascade_frontalface_default.xml"
    eyes_cascade = cv2.CascadeClassifier()
    eyes_cascade_name = pretrained_dir + "/haarcascade_eye.xml"
    
    #eyes_cascade_name = pretrained_dir + "/haarcascade_eye_tree_eyeglasses.xml" #seems to work better in current light conditions
    
    if not face_cascade.load(cv2.samples.findFile(face_cascade_name)):
        print('--(!)Error loading face cascade')
        exit(0)
    if not eyes_cascade.load(cv2.samples.findFile(eyes_cascade_name)):
        print('--(!)Error loading eyes cascade')
        exit(0)

    print("Loaded cascade classifiers")
    return face_cascade, eyes_cascade


# parser = argparse.ArgumentParser(description='Code for Cascade Classifier tutorial.')
# parser.add_argument('--face_cascade', help='Path to face cascade.', default='data/haarcascades/haarcascade_frontalface_alt.xml')
# parser.add_argument('--eyes_cascade', help='Path to eyes cascade.', default='data/haarcascades/haarcascade_eye_tree_eyeglasses.xml')
# parser.add_argument('--camera', help='Camera divide number.', type=int, default=0)
# args = parser.parse_args()
#face_cascade_name = args.face_cascade
#eyes_cascade_name = args.eyes_cascade
def load_png_object(file_path: str):
    im = cv2.imread(file_path,cv2.IMREAD_UNCHANGED)
    im_dims = im.shape 
    print(f"loaded object: {file_path} with shape: {im_dims}")
    return im, im_dims

def insert_hat_on_frame(frame_rgb, obj, coords, anchor):
    frame = frame_rgb.copy()
    
    Hf, Wf = frame.shape[:2]
    Hh, Wh = obj.shape[:2]

    hat_mask = obj[:, :, 3] > 0
    hat_rgb  = obj[:, :, :3]

    # bottom-center align - works when no rotation is present
    # slice_start_y = int(coords[1]) - Hh
    # slice_start_x = int(coords[0]) - Wh // 2

    # coords = where anchor should go in the rgb frame, i.e. face coordinate position
    target_y, target_x = coords
    anchor_y, anchor_x = anchor  
    
    # top-left corner on frame where hat-image should start
    #slice_start_y = #anchor_y-int(target_y) #target_y #int(target_y - anchor_y)
    #slice_start_x = #anchor_x-int(target_x)#target_x #int(target_x + anchor_x)
    slice_start_y = int(target_y - anchor_y)
    slice_start_x = int(target_x - anchor_x)

    # ---- STEP 1: handle negative starts (crop top/left) ----
    crop_x_left = max(0, -slice_start_x)
    crop_y_top  = max(0, -slice_start_y)

    x1 = max(0, slice_start_x)
    y1 = max(0, slice_start_y)

    # ---- STEP 2: compute end positions BEFORE cropping bottom/right ----
    x2 = x1 + (Wh - crop_x_left)
    y2 = y1 + (Hh - crop_y_top)

    # ---- STEP 3: crop bottom/right if going outside frame ----
    crop_x_right = max(0, x2 - Wf)
    crop_y_bottom = max(0, y2 - Hf)

    # Compute final slice ranges AFTER all cropping
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

def time_smoothen_detections(arr_new: np.ndarray, arr_ref: np.ndarray, alpha=0.95) -> np.ndarray:
    return (alpha*arr_ref + (1-alpha)*arr_new).astype(int)
    

def rotate_object(object_im: np.ndarray, angle: float, anchor_point: tuple) -> np.ndarray:
    """_summary_

    Args:
        object_im (np.ndarray): _description_
        angle (float): _description_
        anchor_point (tuple): (y,x) 

    Returns:
        np.ndarray: _description_
    """
    h, w = object_im.shape[0], object_im.shape[1]
    object_im = rotate(object_im, angle, center=(anchor_point[1],anchor_point[0]),order=1, resize=True) #skimage centre is (x,y)
    
    # #we need to know the anchor point post-translation and resizing. Easiest way is to rotate a mask containing the anchor point
    anchor_mask = np.zeros((h, w), dtype=np.uint8)
    anchor_mask[anchor_point] = 1
    anchor_rot = rotate(anchor_mask, angle, center=(anchor_point[1],anchor_point[0]), resize=True, order=0, preserve_range=True)

    # 3) Get new anchor coordinates
    new_anchor = np.argwhere(anchor_rot == 1)[0]   # row, col in rotated space -> y, x, same as output 
    return object_im, new_anchor #new_anchor
    
def rescale_object(object_im: np.ndarray, face_width: float) -> np.ndarray: 
    width_scale_factor = face_width/object_im.shape[1]
    out_im = rescale(object_im, width_scale_factor,channel_axis=-1, order=1)
    return out_im

if __name__=="__main__":
    ###step 1: connect to webcam
    USE_DROID_CAM = False 
    object_im, object_dims = load_png_object("image_props/hat_rescaled.png")
    object_im_trf = object_im.copy()
    cap = connect_camera(USE_DROID_CAM)
    
    ###step 2: load classifiers 
    pretrained_dir = "pretrained_classifiers"
    face_cascade, eyes_cascade = load_cascades(pretrained_dir)
    
    
    
    # To keep track of frames per second
    start_time = time.time()
    font = cv2.FONT_HERSHEY_COMPLEX
    n_frames = 0

    #make the loop
    stop = False
    N_eyes = 0
    
    while not stop:
        ret, frame = cap.read()
        if frame is None:
            print('--(!) No captured frame -- Break!')
            break
            
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) #convert to grayscale
        frame_gray = cv2.equalizeHist(frame_gray) #why do we do this?
        #-- Detect faces
        faces = face_cascade.detectMultiScale(frame_gray) #is a np.array (N,4)
        if isinstance(faces,tuple): 
           N_Faces = 0
        else:
           N_Faces = faces.shape[0] #let them extract this
        
        for (x,y,w,h) in faces:
            
            center = (int(x + w//2), int(y + h//2))
            frame = cv2.ellipse(frame, center, (w//2, h//2), 0, 0, 360, (255, 0, 255), 4)
            faceROI = frame_gray[y:y+h,x:x+w]
            #-- In each face, detect eyes
            eyes = eyes_cascade.detectMultiScale(faceROI)
            N_eyes = len(eyes)
            eye_center_holder = []
            eye_coord_arr = np.zeros((2,2))
    
            for i, (x2,y2,w2,h2) in enumerate(eyes):
                eye_center = (x + x2 + w2//2, y + y2 + h2//2)
                radius = int(round((w2 + h2)*0.25))
                frame = cv2.circle(frame, eye_center, radius, (255, 0, 0 ), 4)
                if N_eyes == 2:
                    eye_coord_arr[i,:] = np.array([eye_center[0],eye_center[1]]) #x, y in global coords
                eye_center_holder.append(eye_center)
            if N_eyes>=2: #case: pair of eyes could be detected
                cv2.line(frame,eye_center_holder[0],eye_center_holder[1],color=(0, 255, 0 ))
                
                
                #compute the line between the two eye coordinates - ensure direction in positive x and use the two eyes which have the maximal horizontal distance

                #step 1: heuristic for left vs right eye 
                right_eye_idx = np.argmax(eye_coord_arr[:,0]) #actually we should use the pair which has the maximal horizontal distance
                
                right_eye = eye_coord_arr[right_eye_idx,:]
                left_eye = eye_coord_arr[1-right_eye_idx,:]


                if left_eye[0] > right_eye[0]:
                    left_eye, right_eye = right_eye, left_eye
    
            
                #vector from left eye to right eye
                vec = right_eye - left_eye
                
                ###find the normal vector - to find the translation we need  
                n_vec = np.array([-vec[1],vec[0]])
                norm = np.linalg.norm(n_vec)
                if norm == 0:  #becomes zero if the same eye is detected twice
                    print("Warning: zero-length eye vector, skipping hat placement")
                    
                    continue  # skip this face
                else:
                    n_vec = n_vec/norm

                    end_point_hat_x = int(center[0] - n_vec[0]*w//2)
                    end_point_hat_y = int(center[1] - n_vec[1]*h//2) #subtract because image coordinate system 
                    
                    cv2.circle(frame,(end_point_hat_x,end_point_hat_y),10, color=(0,255,0),thickness=3, lineType=8, shift=0) #blue circle - top of face 
                    cv2.line(frame,center,(end_point_hat_x,end_point_hat_y),color=(255, 0, 0 )) #green line
                    
                    object_im_trf = rescale_object(object_im, w+100)

                    anchor_point = np.array([object_im_trf.shape[0]-1,object_im_trf.shape[1]//2]) #y,x always 
                    print(f"anchor pt before update: {anchor_point}") 
                        
                    
                    theta = np.arctan2(-vec[1],vec[0]) #is in radians
                    theta *= 180/np.pi #note: when you rotate face right it is negative angle and it works not as intended, when you rotate face left it is positive angle and works as intended
                    # Before inserting
                    
                    object_im_trf, anchor_point_new = rotate_object(object_im_trf, theta, anchor_point) #get the corresponding anchor point after applying rotation
                    print(f"anchor pt before update: {anchor_point}") 
                    
                    
                    #issue is anchor point needs to be transformed properly - I think part of the issue is that we do not handle that the hat image has a new size. 
                    #TODO: draw on paper, we probably need to consider the new image size 
                    #dx = anchor_point[1] - object_im_trf.shape[1]//2
                    #dy = anchor_point[0] - (object_im_trf.shape[0]-1)
                    
                    print(f"anchor pt after update: {anchor_point}") 
                    print(f"end point x, y {end_point_hat_x},{end_point_hat_y}")
                    


                    frame = insert_hat_on_frame(frame, object_im_trf, (end_point_hat_y,end_point_hat_x), anchor_point)
                    
                    #for debugging outputs 
                    cv2.line(frame,(0,0),(anchor_point_new[1],anchor_point_new[0]),color=(0, 255, 0 )) #green line
                    cv2.line(frame,(0,0),(anchor_point[1],anchor_point[0]),color=(0, 0, 255 )) #red line
                    
                    np.savetxt(fname=f"frame_data/{n_frames}_end_point.txt",X=np.array([end_point_hat_x,end_point_hat_y]))
                    np.savetxt(fname=f"frame_data/{n_frames}_anchor_new.txt",X=anchor_point_new)
                    np.savetxt(fname=f"frame_data/{n_frames}_anchor_org.txt",X=anchor_point)
                    
                    cv2.imwrite(filename=f"frame_data/{n_frames}.png",img=frame)
                    
        # Keep track of frames-per-second (FPS)
        n_frames = n_frames + 1
        elapsed_time = time.time() - start_time
        fps = int(n_frames / elapsed_time)

        # Put the information on the image frame: FPS, number of faces, number of eyes 
        str_out = f"fps: {fps}, N_f: {N_Faces}, N_e: {N_eyes}"
        cv2.putText(frame, str_out, (100, 100), font, 1, 255, 1)
        
            
            
        
        
        cv2.imshow('Capture - Face detection', frame)
        

        if cv2.waitKey(1) == ord('q'):
            stop = True


