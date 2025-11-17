from __future__ import print_function
import cv2
import argparse
import time


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
    n_frames = 0

    #make the loop
    stop = False

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
            center = (x + w//2, y + h//2)
            frame = cv2.ellipse(frame, center, (w//2, h//2), 0, 0, 360, (255, 0, 255), 4)
            faceROI = frame_gray[y:y+h,x:x+w]
            #-- In each face, detect eyes
            eyes = eyes_cascade.detectMultiScale(faceROI)
            N_eyes = len(eyes)
    
            for (x2,y2,w2,h2) in eyes:
                eye_center = (x + x2 + w2//2, y + y2 + h2//2)
                radius = int(round((w2 + h2)*0.25))
                frame = cv2.circle(frame, eye_center, radius, (255, 0, 0 ), 4)
        
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


