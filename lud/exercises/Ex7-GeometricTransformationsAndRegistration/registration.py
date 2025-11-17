import numpy as np
import matplotlib.pyplot as plt
import math
from skimage.transform import rotate
from skimage.transform import EuclideanTransform
from skimage.transform import SimilarityTransform
from skimage.transform import matrix_transform
from skimage.transform import warp
from skimage.transform import swirl
from skimage import io
import os
import matplotlib.pyplot as plt
from skimage.util.dtype import img_as_float

def show_comparison(original, transformed, transformed_name):
    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(8, 4), sharex=True,
                                   sharey=True)
    ax1.imshow(original)
    ax1.set_title('Original')
    ax1.axis('off')
    ax2.imshow(transformed)
    ax2.set_title(transformed_name)
    ax2.axis('off')
    io.show()


in_dir = 'data'
im_org_path = os.path.join(in_dir, 'NusaPenida.png')
im_org = io.imread(im_org_path)

# angle in degrees - counter clockwise
rotation_angle = 10

def ex1():
    rotated_img = rotate(im_org, rotation_angle)
    show_comparison(im_org, rotated_img, "Rotated image")
        # Angle is given in degrees.
        # By default, the rotation anchor is set at the CENTER

    # Rotation using origo as center:
    rot_img_origo = rotate(im_org, rotation_angle, center=[0,0])
    show_comparison(im_org, rot_img_origo, "Rotated image with origo at (0,0)")
# ex1()

def ex2():
    rotated_img = rotate(im_org, rotation_angle, mode="reflect")
    show_comparison(im_org, rotated_img, "Rotated image")
        # As for filters, reflect simply reflects pixels from its mirror such, that the
        # image corners are still filled
# ex2()

def ex3():
    rotated_img = rotate(im_org, rotation_angle, mode="wrap")
    show_comparison(im_org, rotated_img, "Rotated image")
        # Appends pixels from opposite side of same axis to corner areas.
# ex3()

def ex4():
    rotated_img = rotate(im_org, rotation_angle, resize=True, mode="constant", cval=100)
    show_comparison(im_org, rotated_img, "Rotated image")
        # Keeps all pixels but resizes image to account for pixels that went out of the
        # border area
# ex4()


def ex5():
    rotated_img = rotate(im_org, rotation_angle, resize=True)
    show_comparison(im_org, rotated_img, "Rotated image")
        # Keeps all pixels but does so doing resizing
        # Boundary pixels still present (=0)
# ex5()

# EXERCISE 6
# angle in radians - counter clockwise
rotation_angle = 10.0 * math.pi / 180.
trans = [10, 20]
tform = EuclideanTransform(rotation=rotation_angle, translation=trans) # NOTICE -> first rotation, then translation
print(tform.params)
    # We see, that rotation is a 3 dimensional matrix, meaning, that
    # the transform utilizes homogeneous coordinates.
        # Here, the angle is given in radians

def ex7():
    transformed_img = warp(im_org, tform)
    show_comparison(im_org, transformed_img, "Affine transformed image using warp function")
        # Keeps all pixels but does so doing resizing
        # Boundary pixels still present (=0)
# ex7()

def ex8():
    transformed_img = warp(im_org, tform.inverse)
    show_comparison(im_org, transformed_img, "Affine transformed image using warp function")
        # Does an inverse transformation, both wrt. rotation and translation
# ex8()

def ex9():
    rotation_angle =15*math.pi/180
    trans = [40,30]
    scaling = 0.6
    simtrans = SimilarityTransform(scale=scaling, rotation=rotation_angle, translation=trans)
    transformed_img = warp(im_org, simtrans)
    show_comparison(im_org, transformed_img, "Affine transformation")
# ex9()

# Ex10
def swirl_transform_test():
    """
        Try the NON-LINEAR Swirl transform.
        strength = Controls how strong the swirling effect is
        radius = Controls how large the radius of the swirl is
        center = Controls the central position of the swirl
    """
    str = 20
    r = 300
    c = [200,200]
    swirl_img = swirl(im_org, strength=str, radius=r,center=c)
    show_comparison(im_org, swirl_img, "Non-linear SWIRL transform")
# swirl_transform_test()

##############################
# Landmark based registration
##############################

src_img = io.imread(os.path.join(in_dir, 'Hand1.jpg'))
dst_img = io.imread(os.path.join(in_dir, 'Hand2.jpg'))

# Ex 11
def viz_overlap():
    blend = 0.5 * img_as_float(src_img) + 0.5 * img_as_float(dst_img)
    io.imshow(blend)
    io.show()
# viz_overlap()

# Exercise 12 and 13
src = np.array( [
        [588,274],
        [328,179],
        [134,398],
        [260,525],
        [613,448]
       ] )
dst = np.array([
    [626,295],
    [384,165],
    [202, 276],
    [278,440],
    [589,451]
    ])
def show_landmarks(img,lmarks):
    plt.imshow(img)
    plt.plot(lmarks[:,0], lmarks[:,1], '.r', markersize=12)
    plt.show()
# show_landmarks(src_img, src)
# show_landmarks(dst_img, dst)

def show_src_target_lmarks(src_marks,target_marks):
    fig, ax = plt.subplots()
    ax.plot(src_marks[:, 0], src_marks[:, 1], '-r', markersize=12, label="Source")
    ax.plot(target_marks[:, 0], target_marks[:, 1], '-g', markersize=12, label="Destination")
    ax.invert_yaxis()
    ax.legend()
    ax.set_title("Landmarks before alignment")
    plt.show()
# show_src_target_lmarks(src,dst)


# Ex 14
def compute_objective_function(src,dst):
    """
        Computes how well the two sets of landmarks are aligned.
    """
    f = ((src - dst)**2).sum()
    print(f'Landmark alignment error F: {f}')
compute_objective_function(src,dst)

def make_transform():
    tform = EuclideanTransform()
    tform.estimate(src, dst) # Function that estimates the optimal alignment
    src_transform = matrix_transform(src, tform.params)
    
    # Show transformed landmarks
    show_src_target_lmarks(src_transform,dst)

    # Make new alignment
    src_aligned = warp(src_img, tform.inverse)
        # NOTICE -> we inverse the transform. Why? Image resampling is done using the
        # inverse mapping (see book) - hence, employ double inverse, hehehe.

    # Show new alignment
    blend_before = 0.5 * img_as_float(src_img) + 0.5 * img_as_float(dst_img)
    blend_after = 0.5 * img_as_float(src_aligned) + 0.5 * img_as_float(dst_img)
    show_comparison(blend_before, blend_after, "Estimated optimal alignment")
make_transform()
