from scipy.ndimage import binary_erosion
from skimage import io, color, morphology
from skimage.morphology import binary_closing, binary_opening, binary_erosion, binary_dilation
from skimage.morphology import disk
import matplotlib.pyplot as plt
import numpy as np
from skimage import measure
from skimage.color import label2rgb
import pydicom as dicom
from scipy.stats import norm, ttest_1samp
from scipy.spatial import distance
import os


def show_comparison(original, modified, modified_name):
    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(8, 4), sharex=True,
                                   sharey=True)
    ax1.imshow(original, cmap="gray", vmin=-200, vmax=500)
    ax1.set_title('Original')
    ax1.axis('off')
    ax2.imshow(modified)
    ax2.set_title(modified_name)
    ax2.axis('off')
    io.show()


in_dir = 'data'
train = ['Training.dcm']
val = ['Validation1.dcm','Validation2.dcm','Validation3.dcm']
val_spleen = ['Validation1_spleen.png','Validation2_spleen.png','Validation3_spleen.png']
test = ['Test1.dcm','Test2.dcm','Test3.dcm']
test_spleen = ['Test1_spleen.png','Test2_spleen.png','Test3_spleen.png']

train_CT = dicom.dcmread(os.path.join(in_dir, train[0]))
train_CT_img = train_CT.pixel_array
# print(train_CT_img.shape, train_CT_img.dtype)
# plt.hist(train_CT_img.ravel(),bins=256)

def ex1():
    io.imshow(train_CT_img, vmin=-50, vmax=200, cmap='gray')
    io.show()
# ex1()

# Read label image
spleen_roi = io.imread(os.path.join(in_dir, 'SpleenROI.png'))
# convert to boolean image
spleen_mask = spleen_roi > 0
spleen_values =train_CT_img[spleen_mask]

def ex2():
    print(f'Mean: {train_CT_img.mean()}, std: {train_CT_img.std()}')
    print(f'*Masked image* Mean: {spleen_values.mean()}, std: {spleen_values.std()}')
    # Ex2: Don't match, which makes sense, as train_CT_img takes bone and bg into account.
    # Mean of 50 and std=15 matches with hypothesis of spleen in region (0,150), as (50-3*std, 50-3*std) = (5, 95)
# ex2()

def ex3():
    plt.hist(spleen_values.ravel(), bins=256)
    plt.show()
        # Spleen is normally distributed
# ex3()

def plot_hist_dist(values, ax, anatomic_class:str=""):
    mu, std = values.mean(), values.std()
    n, bins, patches = ax.hist(values, 60, density=1)
    pdf_spleen = norm.pdf(bins, mu, std)
    ax.plot(bins, pdf_spleen)
    ax.set_xlabel('Hounsfield unit')
    ax.set_ylabel('Frequency')
    ax.set_title(f'{anatomic_class} values in CT scan')
    # plt.show()
    # return ax

def ex4(): # Also ex5 and 6
    annotated_tissues = ['FatROI.png', 'LiverROI.png', 'SpleenROI.png', 'BoneROI.png']
    fig, axes = plt.subplots(1, len(annotated_tissues)+1, figsize=(12,6))
    for i, tissue in enumerate( annotated_tissues ):
        roi = io.imread(os.path.join(in_dir, tissue))
        mask = roi > 0
        tissue_values = train_CT_img[mask]
        plot_hist_dist(tissue_values, axes[0], tissue.replace('ROI.png', ''))
        axes[i+1].imshow(mask)
        axes[i+1].set_title(tissue.replace('ROI.png',''))
        axes[i+1].axis('off')

            # All are normally distributed, with
                # Fat in range (-140, -60)
                # Liver in range (-10, 105)
                # Spleen in range (-5, 100)
                # Bone in range (480, 1000)
            # Similarly this means: fat and bone are easily separable from all classes,
            # yet liver, spleen are not, as these overlap significantly
                # As this is the case, spleen and liver are joint into one class
    plt.tight_layout()
    plt.show()
# ex4()

# Compute means of all ROIs
annotated_tissues = ['FatROI.png', ['LiverROI.png', 'SpleenROI.png'], 'BoneROI.png']
mus, stds = [], []

for tissue in annotated_tissues:
    if not isinstance(tissue,str):
        mask = np.zeros_like(train_CT_img, dtype=bool)

        for t in tissue:
            roi = io.imread(os.path.join(in_dir, t))
            mask |= roi > 0
    else:
        roi = io.imread(os.path.join(in_dir, tissue))
        mask = roi > 0
    tissue_values = train_CT_img[mask]
    mus.append(tissue_values.mean())
    stds.append(tissue_values.std())

# Thresholds
T_bg = -200
T_fat_soft = (mus[0] + mus[1])/2
T_soft_bone= (mus[1] + mus[2])/2

def ex8():
    fat_img = (train_CT_img > T_bg) & (train_CT_img <= T_fat_soft)
    soft_img = (train_CT_img > T_fat_soft) & (train_CT_img <= T_soft_bone)
    bone_img = (train_CT_img > T_soft_bone) 
    label_img = fat_img + 2 * soft_img + 3*bone_img
    image_label_overlay = label2rgb(label_img)
    show_comparison(train_CT_img, image_label_overlay, 'Classification results')
# ex8()
    # As classes are well separated, this was successful.

# Now for PARAMETRIC PIXEL CLASSIFICATION

def plot_hist_of_all_tissues():

    annotated_tissues = ['FatROI.png', 'LiverROI.png', 'SpleenROI.png', 'BoneROI.png']
    fig, axes = plt.subplots(1, 1, figsize=(8,8))
    for i, tissue in enumerate( annotated_tissues ):
        roi = io.imread(os.path.join(in_dir, tissue))
        mask = roi > 0
        tissue_values = train_CT_img[mask]
        plot_hist_dist(tissue_values, axes)
    plt.tight_layout()
    plt.show()
# plot_hist_of_all_tissues()
# Manual intersection inspection:
T_fat_soft = -30
T_soft_bone = 300
# ex8()
    # Just as good results

def ex11():
    mu_fat, std_fat = mus[0], stds[0]
    mu_soft, std_soft = mus[1], stds[1]
    mu_bone, std_bone = mus[2], stds[2]
    T_fat_soft = -1000
    T_soft_bone = -1000
    for val in range(-200, 1000):
        if norm.pdf(val, mu_fat, std_fat) < norm.pdf(val, mu_soft, std_soft) and T_fat_soft == -1000:
            T_fat_soft = val
        elif norm.pdf(val,mu_soft, std_soft) < norm.pdf(val, mu_bone, std_bone) and T_soft_bone == -1000 and T_fat_soft != -1000:
            T_soft_bone = val
            break
    print(f'T_fat_soft = {T_fat_soft}, T_soft_bone = {T_soft_bone}')
        # T_fat_soft = -39, T_soft_bone = 136
            # Quite different to manual allocation
# ex11()

# Now we inspect the spleen and try to classify its pixels

# Spleen in range (-5, 100)
t_1, t_2 = -5, 100
spleen_estimate = (train_CT_img > t_1) & (train_CT_img < t_2)
def ex11number2(img):
    spleen_label_colour = color.label2rgb(img)
    io.imshow(spleen_label_colour)
    plt.title("First spleen estimate")
    io.show()
# ex11number2(spleen_estimate)
    # Not very well segmented, as expected - as lot of soft tissue surrounding it    

def ex12(spleen_estimate2):
    footprint = disk(1)
    closed = binary_closing(spleen_estimate2, footprint)

    footprint = disk(10)
    opened = binary_opening(closed, footprint)
    # opened = binary_erosion(closed, footprint)
    # opened = binary_erosion(opened, footprint)
    # opened = binary_erosion(opened, footprint)
    # opened = binary_erosion(opened, footprint)
    # opened = binary_erosion(opened, footprint)
    ex11number2(opened)
    return opened
# morphed_img = ex12(spleen_estimate)

def ex12again(diagnose=False):
    label_img = measure.label(morphed_img)
    region_props = measure.regionprops(label_img)

    if diagnose:
        x_elons = []
        y_elons = []
        for region in region_props:
            min_row, min_col, max_row, max_col = region.bbox
            w, h = max_col-min_col, max_row-min_row
            elongation_x = (w*h) / w
            elongation_y = (w*h) / h
            x_elons.append(elongation_x)
            y_elons.append(elongation_y)
        plt.scatter(x_elons,y_elons)
        plt.show()

    # Spleen should be elongated by elongation_x, which there is one clear BLOB of...
    T_xmin, T_xmax, T_ymin,T_ymax = 100, 130, 50, 80
    label_img_filter = label_img
    for region in region_props: 
        min_row, min_col, max_row, max_col = region.bbox
        w, h = max_col-min_col, max_row-min_row
        elongation_x = (w*h) / w
        elongation_y = (w*h) / h
        if elongation_x < T_xmin or elongation_x >= T_xmax or elongation_y < T_ymin or elongation_y >= T_ymax:
            for cords in region.coords:
                label_img_filter[cords[0], cords[1]] = 0
    spleen = label_img_filter > 0
    show_comparison(morphed_img, spleen, 'Spleen')
        # Success!
# ex12again() # Ex 12 -> ex 14

def morph_img(bin_img):
    footprint = disk(1)
    closed = binary_closing(bin_img, footprint)

    footprint = disk(10)
    opened = binary_opening(closed, footprint)
    return opened


def spleen_finder(img_path, show=False):

    img = dicom.dcmread(img_path).pixel_array

    # Threshold soft tissue
    t_1, t_2 = -5, 100
    spleen_estimate = (img > t_1) & (img < t_2)

    # Morph to separate BLOBs:
    morphed_img = morph_img(spleen_estimate)

    # Create label image of all BLOBs are retrieve their regional properties
    label_img = measure.label(morphed_img)
    region_props = measure.regionprops(label_img)

    # Retrieve only spleen from binary image
    T_xmin, T_xmax, T_ymin,T_ymax = 100, 130, 50, 80
    label_img_filter = label_img
    for region in region_props: 
        min_row, min_col, max_row, max_col = region.bbox
        w, h = max_col-min_col, max_row-min_row
        elongation_x = (w*h) / w
        elongation_y = (w*h) / h
        if elongation_x < T_xmin or elongation_x >= T_xmax or elongation_y < T_ymin or elongation_y >= T_ymax:
            for cords in region.coords:
                label_img_filter[cords[0], cords[1]] = 0
    spleen = label_img_filter > 0
    if show:
        show_comparison(morphed_img, spleen, 'Spleen')
    return spleen

for i in range(len(val)):
    spleen_finder(os.path.join(in_dir, val[i]))
        # Success all the way through

dice_score = lambda X, Y: 1-distance.dice(X.ravel(), Y.ravel())

def compute_dice(val_imgs, val_masks, test_imgs, test_masks):
    
    val_scores = []
    for val_img_path, val_mask_path in zip(val_imgs, val_masks):
        gt_mask = io.imread(os.path.join(in_dir, val_mask_path))    
        gt_spleen = gt_mask > 0
        pred_spleen = spleen_finder(os.path.join(in_dir, val_img_path))

        # Compute dice
        dice_out = dice_score(pred_spleen, gt_mask)
        val_scores.append(dice_out)

    # Now for test
    test_scores = []
    for test_img_path, test_mask_path in zip(test_imgs, test_masks):
        gt_mask = io.imread(os.path.join(in_dir, test_mask_path))    
        gt_spleen = gt_mask > 0
        pred_spleen = spleen_finder(os.path.join(in_dir, test_img_path), show=True)

        # Compute dice
        dice_out = dice_score(pred_spleen, gt_mask)
        test_scores.append(dice_out)

    print(f'Validation scores: {val_scores}, test scores: {test_scores}')
compute_dice(val, val_spleen, test, test_spleen)
    # Overfitting has occurred == wrong threshold set. Works well for val,
    # fails for 2/3 test images. Does however work very good for the others.


