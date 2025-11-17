import time
import cv2
import numpy as np
from skimage.util import img_as_float
from skimage.util import img_as_ubyte

# Helper function to capture camera in pane
def show_in_moved_window(win_name, img, x, y):
    """
    Show an image in a window, where the position of the window can be given
    """
    cv2.namedWindow(win_name)
    cv2.moveWindow(win_name, x, y)
    cv2.imshow(win_name,img)

# Function controls all the logic.
# Function e.g. converts color image to gray-scale image (ex 2)
def capture_from_camera_and_show_images():
    print("Starting image capture")

    print("Opening connection to camera")
    url = 0
    use_droid_cam = True
    if use_droid_cam:
        url = "http://10.181.227.9:4747/video"
    cap = cv2.VideoCapture(url)
    # cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open camera")
        exit()

    print("Starting camera loop")
    # Get first image
    ret, frame = cap.read()
    # if frame is read correctly ret is True
    if not ret:
        print("Can't receive frame")
        exit()

    # Transform image to gray scale and then to float, so we can do some processing
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # Converts to grayscale
    frame_gray = img_as_float(frame_gray) # Converts to floating point image

    # Set alert, thresh and alpha values
    A = 0.15
    T = 0.1
    alpha = 0.50

    # To keep track of frames per second
    start_time = time.time()
    n_frames = 0
    stop = False
    while not stop:
        ret, new_frame = cap.read() # Capture new image
        if not ret:
            print("Can't receive frame. Exiting ...")
            break

        # Transform image to gray scale and then to float, so we can do some processing
        new_frame_gray = cv2.cvtColor(new_frame, cv2.COLOR_BGR2GRAY)
        new_frame_gray = img_as_float(new_frame_gray) # Convert it to grayscale

        # Compute difference image
        dif_img = np.abs(new_frame_gray - frame_gray) # Compute tot diff pixels between ref and new img

        bin_img = (dif_img > T)

        # Compute % of foreground pixels that changed
        pct_change = np.sum(bin_img)/(dif_img.shape[0] * dif_img.shape[1])

        # Keep track of frames-per-second (FPS)
        n_frames = n_frames + 1
        elapsed_time = time.time() - start_time
        fps = int(n_frames / elapsed_time)

        # Put the FPS on the new_frame
        str_out = f"fps: {fps}"
        font = cv2.FONT_HERSHEY_COMPLEX
        cv2.putText(new_frame, str_out, (100, 100), font, 1, 255, 1)
        cv2.putText(new_frame, f'Image size: ({new_frame.shape[0]}, {new_frame.shape[1]})', (20,470), font, 1, (0, 0, 255), 1)
        cv2.putText(dif_img, f'Change in (pixels, %pixels): ({np.sum(bin_img)}, {pct_change})', (20,470), font, 0.5, 20, 2)

        # Print to frame if pct_change exceeds A
        if pct_change > A:
            print(f"EXCEEDS! pct_change = {pct_change}")
            cv2.putText(new_frame, 'BALARM', (200,200), font, 2, 350, 1)

        # Display the resulting frame
        show_in_moved_window('Input', new_frame, 0, 10)
        show_in_moved_window('Input gray', new_frame_gray, 600, 10)
        show_in_moved_window('Binary image', img_as_float(bin_img) * 255, 1200, 10)
        show_in_moved_window('Difference image', dif_img, 1800, 10)

        # Old frame is updated
        frame_gray = alpha * frame_gray + (1 - alpha) * new_frame_gray
        # frame_gray = new_frame_gray

        if cv2.waitKey(1) == ord('q'):
            stop = True

    print("Stopping image loop")
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    capture_from_camera_and_show_images()