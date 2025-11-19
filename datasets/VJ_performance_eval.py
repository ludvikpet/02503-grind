import cv2
import argparse
import os
from skimage import io
import matplotlib.pyplot as plt
import sys
from loguru import logger

logger.remove()
logger.add(sys.stderr, format="<level>{level}</level> | <level>{message}</level>")

def parse_args():
    """ Parse CLI arguments. """

    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True, help='Path to trained classifier')
    parser.add_argument('--input', type=str, required=True, help='Path to media to be analyzed (location: root)')
    parser.add_argument('--type', type=str, help='Note type of media (image/video)', default='image')
    parser.add_argument('--w', type=int, help='Width of trained integral image', default=24)
    parser.add_argument('--h', type=int, help='height of trained integral image', default=24)

    args = parser.parse_args()
    # cwd = os.getcwd()
    # ext = args.model if 'caltech' in os.path.basename(cwd) else os.path.join('caltech-101', args.model)
    # path_to_dir = os.path.join(cwd, ext)

    return args.model, args.input, args.w, args.h, args.type

def load_trained_cascade(cascade_path: str) -> cv2.CascadeClassifier:
    """
        Loads trained cascade classifier from specified directory. Notice: cascade must
        be named cascade.xml.

        :param str cascade_directory:           Path to classifier directory.
    """

    vj_classifier = cv2.CascadeClassifier()
    # vj_classifier_name = os.path.join(cascade_directory, 'cascade.xml')
    
    if not vj_classifier.load(cv2.samples.findFile(cascade_path)):
        logger.error('--(!)Error loading object cascade')
        exit(0)

    logger.debug("Loaded cascade classifier")
    return vj_classifier

def img_object_detection(in_path, w_window, h_window, bbox_color, thickness, annotated_dir):
    """
        Perform object detection using video stream as input.
        
        :param str in_path:                     Path to video.
        :param int w_window:                    Width of detector's image window.
        :param int h_window:                    Height of detector's image window.
        :param Tuple[int,int,int] bbox_color:   Color of bounding box.
        :param int thickness:                   bbox boundary thickness.
        :param str annotated_dir:               Path to annotation output directory.
    """

    # Read image and convert to grayscale
    input_img = io.imread(in_path) # At root
    gray_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY)

    # Compute objects and display on original image if present
    objects = vj_classifier.detectMultiScale(gray_img)

    # Loop over identified objects and draw their respective bounding boxes
    for idx, (x0,y0,w,h) in enumerate(objects):
        logger.debug('Found object! Now drawing bbox...',file=sys.stderr)
        x1, y1 = x0 + w, y0 + h

        # Retrieve bbox-confined image and resize to 24x24 and output to file to conduct further experiments:
        img_bbox_slice = input_img[y0:y1, x0:x1, :]
        resized_slice = cv2.resize(img_bbox_slice, (w_window,h_window), interpolation = cv2.INTER_AREA)
        io.imsave(os.path.join(annotated_dir, f'test_window_{idx}.jpg'), resized_slice)
        logger.debug(f'Saved resized annotation of (w={w_window}, h={h_window}) for further experimentation',file=sys.stderr)

        # Draw bounding box onto input image
        input_img = cv2.rectangle(input_img, (x0,y0), (x1,y1), bbox_color, thickness)
    
    # Show output image
    io.imshow(input_img)
    plt.show() 

def vid_object_detection(in_path, bbox_color, thickness, annotated_dir):
    """
        Perform object detection using video stream as input.
        
        :param str in_path:                     Path to video.
        :param Tuple[int,int,int] bbox_color:   Color of bounding box.
        :param int thickness:                   bbox boundary thickness.
        :param str annotated_dir:               Path to annotation output directory.
    """

    cap = cv2.VideoCapture(in_path)
    font = cv2.FONT_HERSHEY_COMPLEX
    save_annotation = False
    f_idx = 0

    while cap.isOpened():
        ret, frame = cap.read()

        if not ret:
            logger.info("Can't receive frame (stream may have ended). Exiting...")
            break

        # Convert image to grayscale and detect objects of frame
        gray_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        scale_factor = 1.1
        objects = vj_classifier.detectMultiScale(gray_img, scaleFactor=scale_factor)

        if isinstance(objects,tuple): 
            n_objects = 0
        else:
            n_objects = objects.shape[0] #let them extract this

            # Loop over identified objects and draw their respective bounding boxes
            for idx, (x0,y0,w,h) in enumerate(objects):
                x1, y1 = x0 + w, y0 + h

                # Draw bounding box onto input image
                input_img = cv2.rectangle(frame, (x0,y0), (x1,y1), bbox_color, thickness)

        # Save annotation for investigation
        if save_annotation:
            io.imsave(os.path.join(annotated_dir, f'test_window_{f_idx}.jpg'), cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        # Write n_objects onto frame and display
        str_out = f'N_obj={n_objects}'
        cv2.putText(frame, str_out, (100, 100), font, 1, 255, 1)
        cv2.imshow('Video Capture - Object Detection', frame)
        f_idx+=1

        # Break if q is pressed
        if cv2.waitKey(1) == ord('q'):
            break

    # Finish video capture
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    
    # Retrieve path to classifier and load
    path_to_model, in_path, w_window, h_window, in_type = parse_args()
    vj_classifier: cv2.CascadeClassifier = load_trained_cascade(path_to_model)

    # Set bounding box color and its border thickness
    bbox_color = (0,255,0) # Green
    thickness = 2
    annotated_dir = 'annotated_dir'
    os.makedirs(annotated_dir, exist_ok=True)

    # Perform object detection using either an image-, or video as input
    if in_type == 'image':
        img_object_detection(in_path, w_window, h_window, bbox_color, thickness, annotated_dir)

    elif in_type == 'video':
        vid_object_detection(in_path, bbox_color, thickness, annotated_dir)
    else:
        raise ValueError(f'File type not recognized - please specify type (--type [image|video])')

