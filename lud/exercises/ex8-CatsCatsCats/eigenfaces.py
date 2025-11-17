from skimage import io
from skimage.util import img_as_ubyte
import matplotlib.pyplot as plt
import numpy as np
import glob
from sklearn.decomposition import PCA
from skimage.transform import SimilarityTransform
from skimage.transform import warp
import os
import pathlib
from preprocess import preprocess_all_cats, preprocess_one_cat, create_u_byte_image_from_vector

cat_dir = os.getcwd()+'/training_data/'

preprocessed_cats = glob.glob(os.getcwd()+'/preprocessed_cats/*.jpg')
n_samples = len(preprocessed_cats)
model_cat = io.imread(preprocessed_cats[0])
h,w,c = model_cat.shape[0], model_cat.shape[1], model_cat.shape[2]
n_features = h*w*c

# Load data matrix
if os.path.exists('data/data_matrix.npy'):
    data_matrix = np.load('data/data_matrix.npy')
else:
    data_matrix = np.zeros((n_samples, n_features))

    for i, cat_path in enumerate(preprocessed_cats):
        preprocessed_cat = io.imread(cat_path)
        data_matrix[i, :] = preprocessed_cat.flatten()
    np.save('data/data_matrix.npy', data_matrix)

missing_cat_pp = io.imread('data/MissingCatProcessed.jpg')
flat_cat = missing_cat_pp.flatten()

# Compute SSD from reference missing cat image to dataset
sub_data = data_matrix - flat_cat
sub_distances = np.linalg.norm(sub_data, axis = 1)

# print(f"Closest cat to the missing cat: {np.argmin(sub_distances)}")
print(sub_distances.shape)
