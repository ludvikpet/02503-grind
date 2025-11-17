from skimage import color
from skimage.util import img_as_ubyte
from skimage.util import img_as_float
import time
import cv2


def show_in_moved_window(win_name, img, x, y):
    """
    Show an image in a window, where the position of the window can be given
    """
    cv2.namedWindow(win_name)
    cv2.moveWindow(win_name, x, y)
    cv2.imshow(win_name, img)


def process_gray_image(img):
    """
    Do a simple processing of an input gray scale image and return the processed image.
    # https://scikit-image.org/docs/stable/user_guide/data_types.html#image-processing-pipeline
    """
    img_float = img_as_float(img)
    img_proc = 1 - img_float #invert image, meaning light becomes dark and vice versa 
    return img_as_ubyte(img_proc)

def threshold_gray_image(img, thresh = 0.5):
    mask = img > thresh
    return (mask * 255).astype("uint8")


def detect_dtu_signs_hsv(img):
    img_hsv = color.rgb2hsv(img)
    h_comp = img_hsv[:, :, 0]
    v_comp = img_hsv[:, :, 2]
    
    mask = (h_comp > 0.90) & (v_comp>=0.478) & (v_comp < 0.70) 

    return (mask*255).astype("uint8") 

def process_rgb_image(img):
    """
    Simple processing of a color (RGB) image
    """
    # Copy the image information so we do not change the original image
    proc_img = img.copy()
    r_comp = proc_img[:, :, 0]
    proc_img[:, :, 0] = 1 - r_comp #invert r channel, meaning low red before becomes high and high red before becomes low. I.e. green and blue become red, whereas red becomes dark
    return proc_img


def capture_from_camera_and_show_images(process_rgb=False, thresh=False, detect_dtu = False):
    print("Starting image capture")

    print("Opening connection to camera")
    url = 0
    use_droid_cam = False
    if use_droid_cam:
        url = "http://192.168.1.120:4747/video"
    cap = cv2.VideoCapture(url)
    if not cap.isOpened():
        print("Cannot open camera")
        exit()

    print("Starting camera loop")
    # To keep track of frames per second using a high-performance counter
    old_time = time.perf_counter()
    fps = 0
    stop = False
    font = cv2.FONT_HERSHEY_COMPLEX
    threshval = 0.5
    while not stop:
        ret, new_frame = cap.read()
        if not ret:
            print("Can't receive frame. Exiting ...")
            break

        # Change from OpenCV BGR to scikit image RGB
        new_image = new_frame[:, :, ::-1]
        new_image_gray = color.rgb2gray(new_image) #this is float
        if process_rgb:
            proc_img = process_rgb_image(new_image)
            # convert back to OpenCV BGR to show it
            proc_img = proc_img[:, :, ::-1]
        elif thresh:
            proc_img = threshold_gray_image(new_image_gray,threshval)
            cv2.putText(proc_img, f"{threshval:.2f}", (100, 100), font, 1, 255, 1)
        elif detect_dtu: 
            proc_img = detect_dtu_signs_hsv(new_image)
        else:
            proc_img = process_gray_image(new_image_gray) 

        # update FPS - but do it slowly to avoid fast changing number
        new_time = time.perf_counter()
        time_dif = new_time - old_time
        old_time = new_time
        fps = fps * 0.95 + 0.05 * 1 / time_dif

        # Put the FPS on the new_frame
        str_out = f"fps: {int(fps)}"
        cv2.putText(new_frame, str_out, (100, 100), font, 1, 255, 1)

        # Display the resulting frame
        show_in_moved_window('Input', new_frame, 0, 10)
        show_in_moved_window('Input gray', new_image_gray, 600, 10)
        show_in_moved_window('Processed image', proc_img, 1200, 10)

        key = cv2.waitKey(1)

        if key == ord('q'):
            stop = True
        if key == ord("j"):
            threshval -= 0.05
        if key == ord("k"):
            threshval += 0.05
        


        

    print("Stopping image loop")
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    capture_from_camera_and_show_images(detect_dtu = True)
