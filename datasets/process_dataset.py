import os
from scipy.io import loadmat
from skimage import io
from glob import glob
import argparse
from skimage.util import img_as_float, img_as_ubyte
from skimage.color import rgb2gray, rgb2hsv


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pos', type=str, help='Explicit name of positive dataset directory')
    parser.add_argument('--neg', type=str, help='Explicit name of negative dataset directory')
    args = parser.parse_args()
    return args.pos, args.neg

def write_to_grayscale(img_path, out_path):
    """
        Format image to grayscale and save to output directory
    """
    img = io.imread(img_path)
    if img.ndim == 3 and img.shape[-1] > 1:
        img = rgb2gray(img)
    img = img_as_ubyte(img)

    img_name = os.path.basename(img_path)
    gray_path = os.path.join(out_path, img_name)
    io.imsave(gray_path,arr=img)

    # Slice gray path for further use
    gray_path = gray_path.replace('caltech-101/', "") 
    print(gray_path)

    return gray_path

def preprocess_data(data_dir: str, pos_class: str, neg_class: str) -> None:

    # Retrieve img and annotation directories for classes
    img_dir = os.path.join(data_dir,f'101_ObjectCategories/{pos_class}')
    neg_img_dir = os.path.join(data_dir,f'101_ObjectCategories/{neg_class}')
    annotation_dir = os.path.join(data_dir,f'Annotations/{pos_class}')

    # Make out dirs:
    pos_out = os.path.join(data_dir,f'{pos_class}_gray')
    neg_out = os.path.join(data_dir,f'{neg_class}_gray')
    os.makedirs(neg_out, exist_ok=True)
    os.makedirs(pos_out, exist_ok=True)

    # Get annotation and image files 
    annotation_files = sorted(glob(annotation_dir+'/*'))
    img_files = sorted(glob(img_dir+'/*'))
    neg_img_files = glob(neg_img_dir+"/*")

    # Cut negative class dataset size to fit the positive subset
    N = len(img_files)
    print(neg_img_dir)
    neg_img_files = sorted((neg_img_files + [""] * N)[:N], key = lambda x: (x == "", x))
    print(neg_img_files)

    
    # Create variables for line retrieval
    positive = []
    negative = []

    for idx, (file, neg_file) in enumerate(zip(annotation_files,neg_img_files)):

        # Retrieve bbox
        box_coord = loadmat(file)['box_coord'][0] # Current format: y0,y1,x0,x1

        # Reformat box coordinate to expected format: (x0,y0,w,h)
        w, h = box_coord[3] - box_coord[2],box_coord[1] - box_coord[0]
        box_str = f'{box_coord[2]} {box_coord[0]} {w} {h}'

        # Get positive and negative img file and write
        img_file = img_files[idx]
        gray_file = write_to_grayscale(img_file, out_path=pos_out) 
        positive.append(f"{gray_file} 1 {box_str}")
        if neg_file != "":
            neg_gray_file = write_to_grayscale(neg_file, out_path=neg_out)
            negative.append(neg_gray_file)


    # Write to files
    with open(os.path.join(data_dir,f'{pos_class}_pos.dat'), 'w+') as f:
        f.write('\n'.join(positive))
    with open(os.path.join(data_dir, f'{pos_class}_neg.dat'), 'w+') as f:
        f.write('\n'.join(negative))

if __name__ == '__main__':
    data_dir = 'caltech-101'
    pos_class, neg_class = parse_args()
    
    print(f'Now processing with classes (positive,negative): ({pos_class},{neg_class})')
    preprocess_data(data_dir, pos_class, neg_class)
