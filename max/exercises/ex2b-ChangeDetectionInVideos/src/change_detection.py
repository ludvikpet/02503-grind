import time
import cv2
import numpy as np
from skimage.util import img_as_float
from skimage.util import img_as_ubyte
from loguru import logger 


def show_in_moved_window(win_name, img, x, y):
    """
    Show an image in a window, where the position of the window can be given
    """
    cv2.namedWindow(win_name)
    cv2.moveWindow(win_name, x, y)
    cv2.imshow(win_name,img)

def camera_connect() -> cv2.VideoCapture:
    logger.info("Setting up camera stream...")
    cap = cv2.VideoCapture(0)
    if not cap.isOpened(): 
        logger.error("Couldn't establish connection to camera.")
        exit("Aborted")
    return cap 
    
def read_and_convert_frame(cap: cv2.VideoCapture) -> tuple[np.ndarray,np.ndarray]:
    """
    Reading and conversion pipeline from a VideoCapture object. 
    Conversion of RGB frame to grayscale and floating point image

    Args:
        cap (cv2.VideoCapture): capture device object

    Returns:
        tuple(np.ndarray,np.ndarray): raw unprocessed image, grayscale frame in float format
    
    """
    #read image
    ret, frame = cap.read()
    if not ret:
        logger.error("Couldn't read from camera")
        exit()
    #convert image
    frame_gray = cv2.cvtColor(src=frame,code=cv2.COLOR_BGR2GRAY)
    frame_gray = img_as_float(frame_gray)
    return frame, frame_gray     
    

def change_detection_pipeline(alpha: float, T: float, A: float) -> None:
    cap = camera_connect()
    _, background_im = read_and_convert_frame(cap)
    IM_H, IM_W = background_im.shape[0], background_im.shape[1]
    N_pix = IM_H*IM_W
    font = cv2.FONT_HERSHEY_COMPLEX

    start_time = time.time()
    stop = False 
    n_frames = 0
    while not stop: 
        RGB_frame, new_frame = read_and_convert_frame(cap)
        
        diff_img = np.abs(new_frame - background_im)
        thresh_mask = diff_img>T
        N_foreground_pix = thresh_mask.ravel().sum()
        foreground_frac = N_foreground_pix/N_pix
        #threshold op
        diff_img[:,:]= 0.0 #we expect foreground small, so it is cheaper to zero-fill and use mask for foreground fill 
        diff_img[thresh_mask] = 1.0
        thresh_mask = img_as_ubyte(thresh_mask) #as per specifications
        

        n_frames += 1 
        fps = int(n_frames/(time.time()-start_time)) 
        
    

        #ALERT
        if foreground_frac>A:
            logger.info("CHANGE DETECTED") #this part works
            cv2.putText(RGB_frame, f"fps: {fps} | CHANGE DETECTED", (100, 100), font, 1, (1,1,255), 1)
        else:
            cv2.putText(RGB_frame,f"fps: {fps}", (100,100), font, 1, (255,255,255), 1)

        foreground_info_str = f"N_F:F_R = {N_foreground_pix}/{foreground_frac:.2f}"
        cv2.putText(diff_img,foreground_info_str, (100,100), font, 1, 255, 1)
        
        #visualizations 
        show_in_moved_window('Input', RGB_frame, 0, 10)
        show_in_moved_window('Background image', background_im, 600, 10)
        show_in_moved_window('Difference image', diff_img, 1200, 10)
        show_in_moved_window('Binary image', thresh_mask, 1800, 10)
        #update background
        background_im = alpha*background_im + (1-alpha)*new_frame 

        if cv2.waitKey(1) == ord("q"):
            stop = True 
    logger.info("Stopped imaging")
    cap.release()
    cv2.destroyAllWindows() 
        
        

        

if __name__=="__main__":
    #read sys args 
    A = 0.05
    alpha = 0.95 
    T = 0.1
    change_detection_pipeline(alpha,T,A)