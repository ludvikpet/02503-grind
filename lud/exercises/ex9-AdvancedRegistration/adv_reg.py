import numpy as np
import matplotlib.pyplot as plt
import SimpleITK as sitk
from IPython.display import clear_output
from skimage.util import img_as_ubyte
import os

def imshow_orthogonal_view(sitkImage, origin = None, title=None):
    """
    Display the orthogonal views of a 3D volume from the middle of the volume.

    Parameters
    ----------
    sitkImage : SimpleITK image
        Image to display.
    origin : array_like, optional
        Origin of the orthogonal views, represented by a point [x,y,z].
        If None, the middle of the volume is used.
    title : str, optional
        Super title of the figure.

    Note:
    On the axial and coronal views, patient's left is on the right
    On the sagittal view, patient's anterior is on the left
    """
    data = sitk.GetArrayFromImage(sitkImage)

    if origin is None:
        origin = np.array(data.shape) // 2

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    data = img_as_ubyte(data/np.max(data))
    axes[0].imshow(data[origin[0], ::-1, ::-1], cmap='gray')
    axes[0].set_title('Axial')

    axes[1].imshow(data[::-1, origin[1], ::-1], cmap='gray')
    axes[1].set_title('Coronal')

    axes[2].imshow(data[::-1, ::-1, origin[2]], cmap='gray')
    axes[2].set_title('Sagittal')

    [ax.set_axis_off() for ax in axes]

    if title is not None:
        fig.suptitle(title, fontsize=16)
    plt.show()

def overlay_slices(sitkImage0, sitkImage1, origin = None, title=None):
    """
    Overlay the orthogonal views of a two 3D volume from the middle of the volume.
    The two volumes must have the same shape. The first volume is displayed in red,
    the second in green.

    Parameters
    ----------
    sitkImage0 : SimpleITK image
        Image to display in red.
    sitkImage1 : SimpleITK image
        Image to display in green.
    origin : array_like, optional
        Origin of the orthogonal views, represented by a point [x,y,z].
        If None, the middle of the volume is used.
    title : str, optional
        Super title of the figure.

    Note:
    On the axial and coronal views, patient's left is on the right
    On the sagittal view, patient's anterior is on the left
    """
    vol0 = sitk.GetArrayFromImage(sitkImage0)
    vol1 = sitk.GetArrayFromImage(sitkImage1)

    if vol0.shape != vol1.shape:
        raise ValueError('The two volumes must have the same shape.')
    if np.min(vol0) < 0 or np.min(vol1) < 0: # Remove negative values - Relevant for the noisy images
        vol0[vol0 < 0] = 0
        vol1[vol1 < 0] = 0
    if origin is None:
        origin = np.array(vol0.shape) // 2

    sh = vol0.shape
    R = img_as_ubyte(vol0/np.max(vol0))
    G = img_as_ubyte(vol1/np.max(vol1))

    vol_rgb = np.zeros(shape=(sh[0], sh[1], sh[2], 3), dtype=np.uint8)
    vol_rgb[:, :, :, 0] = R
    vol_rgb[:, :, :, 1] = G

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    axes[0].imshow(vol_rgb[origin[0], ::-1, ::-1, :])
    axes[0].set_title('Axial')

    axes[1].imshow(vol_rgb[::-1, origin[1], ::-1, :])
    axes[1].set_title('Coronal')

    axes[2].imshow(vol_rgb[::-1, ::-1, origin[2], :])
    axes[2].set_title('Sagittal')

    [ax.set_axis_off() for ax in axes]

    if title is not None:
        fig.suptitle(title, fontsize=16)
    plt.show()

def composite2affine(composite_transform, result_center=None):
    """
    Combine all of the composite transformation's contents to form an equivalent affine transformation.
    Args:
        composite_transform (SimpleITK.CompositeTransform): Input composite transform which contains only
                                                            global transformations, possibly nested.
        result_center (tuple,list): The desired center parameter for the resulting affine transformation.
                                    If None, then set to [0,...]. This can be any arbitrary value, as it is
                                    possible to change the transform center without changing the transformation
                                    effect.
    Returns:
        SimpleITK.AffineTransform: Affine transformation that has the same effect as the input composite_transform.
    
    Source:
        https://github.com/InsightSoftwareConsortium/SimpleITK-Notebooks/blob/master/Python/22_Transforms.ipynb
    """
    # Flatten the copy of the composite transform, so no nested composites.
    flattened_composite_transform = sitk.CompositeTransform(composite_transform)
    flattened_composite_transform.FlattenTransform()
    tx_dim = flattened_composite_transform.GetDimension()
    A = np.eye(tx_dim)
    c = np.zeros(tx_dim) if result_center is None else result_center
    t = np.zeros(tx_dim)
    for i in range(flattened_composite_transform.GetNumberOfTransforms() - 1, -1, -1):
        curr_tx = flattened_composite_transform.GetNthTransform(i).Downcast()
        # The TranslationTransform interface is different from other
        # global transformations.
        if curr_tx.GetTransformEnum() == sitk.sitkTranslation:
            A_curr = np.eye(tx_dim)
            t_curr = np.asarray(curr_tx.GetOffset())
            c_curr = np.zeros(tx_dim)
        else:
            A_curr = np.asarray(curr_tx.GetMatrix()).reshape(tx_dim, tx_dim)
            c_curr = np.asarray(curr_tx.GetCenter())
            # Some global transformations do not have a translation
            # (e.g. ScaleTransform, VersorTransform)
            get_translation = getattr(curr_tx, "GetTranslation", None)
            if get_translation is not None:
                t_curr = np.asarray(get_translation())
            else:
                t_curr = np.zeros(tx_dim)
        A = np.dot(A_curr, A)
        t = np.dot(A_curr, t + c - c_curr) + t_curr + c_curr - c

    return sitk.AffineTransform(A.flatten(), t, c)

# Callback invoked when the StartEvent happens, sets up our new data.
def start_plot():
    global metric_values, multires_iterations
    
    metric_values = []
    multires_iterations = []

# Callback invoked when the EndEvent happens, do cleanup of data and figure.
def end_plot():
    global metric_values, multires_iterations
    
    del metric_values
    del multires_iterations
    # Close figure, we don't want to get a duplicate of the plot latter on.
    plt.close()

# Callback invoked when the IterationEvent happens, update our data and display new figure.
def plot_values(registration_method):
    global metric_values, multires_iterations
    
    metric_values.append(registration_method.GetMetricValue())                                       
    # Clear the output area (wait=True, to reduce flickering), and plot current data
    clear_output(wait=True)
    # Plot the similarity metric values
    plt.plot(metric_values, 'r')
    plt.plot(multires_iterations, [metric_values[index] for index in multires_iterations], 'b*')
    plt.xlabel('Iteration Number',fontsize=12)
    plt.ylabel('Metric Value',fontsize=12)
    plt.show()
    
# Callback invoked when the sitkMultiResolutionIterationEvent happens, update the index into the 
# metric_values list. 
def update_multires_iterations():
    global metric_values, multires_iterations
    multires_iterations.append(len(metric_values))

def command_iteration(method):
    print(
        f"{method.GetOptimizerIteration():3} "
        + f"= {method.GetMetricValue():10.5f} "
        + f": {method.GetOptimizerPosition()}"
    )

########################################################################################
# EXERCISES
########################################################################################

# Read 3D image using SITK
vol_sitk = sitk.ReadImage(os.path.join('data', 'ImgT1.nii'))

# Ex 1: Show orthogonal view of CT scan
def ex1():
    imshow_orthogonal_view(vol_sitk, title='T1.nii')
# ex1()

# Ex 2
def rotation_matrix(pitch, roll, yaw):
    pitch, roll, yaw = np.deg2rad(pitch), np.deg2rad(roll), np.deg2rad(yaw)
    R_x = np.array([
        [1,0,0,0],
        [0, np.cos(pitch), -np.sin(pitch), 0],
        [0, np.sin(pitch), np.cos(pitch),0],
        [0,0,0,1]]
    )
    R_y = np.array([
        [np.cos(roll),0,np.sin(roll),0],
        [0, 1, 0, 0],
        [-np.sin(roll), 0, np.cos(roll),0],
        [0,0,0,1]]
    )
    R_z = np.array([
        [np.cos(yaw),-np.sin(yaw),0,0],
        [np.sin(yaw), np.cos(yaw),0,0],
        [0, 0, 1, 0],
        [0,0,0,1]]
    )
    A_R = R_x @ R_y @ R_z
    return A_R

centre_image = np.array(vol_sitk.GetSize()) / 2 - 0.5 # Image coordinate system
centre_world = vol_sitk.TransformContinuousIndexToPhysicalPoint(centre_image) # World coordinate system
def construct_affine_matrix(R, t=None, S=None, z=None):
    if not t:
        t = np.zeros(3)
    if not S:
        S = np.eye(4)
    if not z:
        z = np.ones((4,4)) - np.eye(4)
        z[3, :3] = 0
        z[:3, 3] = 0
    # A = np.eye(4)
    A = sitk.AffineTransform(3)
    # A = R*A*S*z
    # A[:3, 3] = t
    A.SetCenter(centre_world)
    A.SetMatrix(R[:3,:3].T.flatten())
    return A
rot_matrix = rotation_matrix(pitch=25, roll=0, yaw=0)
A_transform = construct_affine_matrix(rot_matrix)

# Ex 3
ImgT1_A = sitk.Resample(vol_sitk, A_transform)
sitk.WriteImage(ImgT1_A, os.path.join('data', 'ImgT1_A.nii'))

# Ex4
def ex4():
    imshow_orthogonal_view(ImgT1_A, title='Transformed ImgT1')
    overlay_slices(vol_sitk, ImgT1_A, title = 'ImgT1 (red) vs. ImgT1_A (green)')
# ex4()

################################################################################
# Ex 5 -> perform registration of moving image to fixed image
################################################################################

# Redefine images for notational purposes
src = ImgT1_A # Moving image
dst = vol_sitk # Fixed image
def print_transform_statistics(tform_reg):
    """Look at statistics of final transform:"""
    estimated_tform = tform_reg.GetNthTransform(0).GetMatrix() # Transform matrix
    estimated_translation = tform_reg.GetNthTransform(0).GetTranslation() # Translation vector
    params = tform_reg.GetParameters() # Parameters (Rx, Ry, Rz, Tx, Ty, Tz)
    print(f'Estimated transform:\n{estimated_tform}\nEstimated translation:\n{estimated_translation}\nParameters:\n{params}')

def perform_registration(src, dst,step_size=0.1, with_composite=False, plot=True, write=True, name='',noise_robust=False):
    R = sitk.ImageRegistrationMethod() # This is the registration pipeline (as seen in transition diagram)

    # Set level of the pyramid schedule (pyramid step)
    if noise_robust:
        R.SetShrinkFactorsPerLevel(shrinkFactors=[2,2,2]) # Downscale image by n
        R.SetSmoothingSigmasPerLevel(smoothingSigmas=[3,1,0]) # No smoothing
        R.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
            # What this means:
                # We have three levels of registration -> shrink factor = 2 + smoothing sigma=3 ; SF=2 + SS=1 ; SF=2 + SF=0 (no smoothing)
                # Shrink factor is not cumulative - we keep using the same.
                # This means, that by employing a larger smoothing at the start, we can make an initial rough
                # estimate of the real offset. Then later on, we can remove the smoothing (and in other cases
                # the shrink factor) to finish the optimization.
    else:
        n = 4
        R.SetShrinkFactorsPerLevel(shrinkFactors=[4]) # Downscale image by n
        R.SetSmoothingSigmasPerLevel(smoothingSigmas=[0]) # No smoothing
        R.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()

    # Set interpolator
    R.SetInterpolator(sitk.sitkLinear) # Linear interpolator

    # Set the similarity metric
    R.SetMetricAsMeanSquares() # This should do the trick, as intensity specification of images are equal
    # R.SetMetricAsJointHistogramMutualInformation() # Try this later on

    # Set the sampling strategy
    R.SetMetricSamplingStrategy(R.RANDOM) # Use stochastic sampling to simplify problem - find random coordinate as well
    R.SetMetricSamplingPercentage(0.50) # We sample 50% of the image

    # Set the optimizer
    R.SetOptimizerAsPowell(stepLength=step_size, numberOfIterations=25) # What is Powell again? Try SGD as well

    # Initialize the transformation type as RIGID
    # initTransform = sitk.Euler3DTransform()
        # Using this initial transform, we're initializing a much more difficult optimization
        # Problem than necessary, as rotation is made by default using the fixed image's
        # origin as the rotation center
    if with_composite:
        # Composite for 240 -> 0: 2x 60
        
        tform_60, tform_180, tform_240 = [sitk.ReadTransform(os.path.join('data', f'A_{gamma}.tfm')) for gamma in ['60', '180', '240']]
        tform_0 = sitk.ReadTransform(os.path.join('data','A1.tfm'))
        tform_composite = sitk.CompositeTransform(3)

        # tform_composite.AddTransform(tform_240.GetNthTransform(0)) 
        # tform_composite.AddTransform(tform_180.GetNthTransform(0))
        tform_composite.AddTransform(tform_60.GetNthTransform(0))
        tform_composite.AddTransform(tform_60.GetNthTransform(0))
        # tform_composite.AddTransform(tform_0.GetNthTransform(0))
        # Transform the composite transform to an affine transform
        affine_composite = composite2affine(tform_composite, centre_world)
        # img240 = sitk.ReadImage(os.path.join('data', 'ImgT1_240.nii'))
        overlay_slices(src,sitk.Resample(src, affine_composite), title='Did the composite transform work?')
        imshow_orthogonal_view(dst, title='The destination')
        initTransform = affine_composite
    else:
        initTransform = sitk.CenteredTransformInitializer(dst, src, sitk.Euler3DTransform(), sitk.CenteredTransformInitializerFilter.GEOMETRY)
            # Changing the rotation axis to the center of the image relaxes the problem significantly,
            # taking only 2 iterations to finish pose estimation
    R.SetInitialTransform(initTransform, inPlace=False)

    if plot:
        # Extra functions to evaluate performance on-the-fly
        R.AddCommand(sitk.sitkStartEvent, start_plot) # Plot the similarity metric values across iterations
        R.AddCommand(sitk.sitkEndEvent, end_plot)
        R.AddCommand(sitk.sitkMultiResolutionIterationEvent, update_multires_iterations)
        R.AddCommand(sitk.sitkIterationEvent, lambda: plot_values(R))

    # Estimate the registration transformation [metric, optimizer, transform]
    tform_reg = R.Execute(dst, src) # Transform
    

    # Apply the estimated transformation to the moving average image
    ImgT1_B = sitk.Resample(src, tform_reg) # Apply transform
    # Save image
    
    if write:
        # Write transform to file
        print(name)
        filename = 'A1.tfm' if not name else f'A_{name.split('_')[1].split('.')[0]}.tfm'
        tform_reg.WriteTransform(os.path.join('data', filename))

        # Write image to file
        filename = 'ImgT1_B.nii' if not name else name
        sitk.WriteImage(ImgT1_B, os.path.join('data', filename))

    # Print final metric value:
    print(f'Metric output: {R.GetMetricValue()}')
    
    # Look at statistics of final transform:
    print_transform_statistics(tform_reg)
    overlay_slices(dst, ImgT1_B, title = 'Fixed image (red) vs. transformed moving image (green)')

# perform_registration(src,dst,plot=True, write=True)

#Output image
res_img = sitk.ReadImage(os.path.join('data', 'ImgT1_B.nii'))
def ex6():
    imshow_orthogonal_view(res_img, title='Transformed ImgT1')
    overlay_slices(vol_sitk, res_img, title = 'ImgT1 (red) vs. transformed back ImgT1_B (green)')
# ex6()
    # Perfectly aligned. This is expected for several reasons:
        # 1) Optimization problem is quite simple, as only rotation is applied
        # 2) Given that the intensity values are equal for the two images, MSE is a valid cost function to use for estimation

# Read transform matrix
def ex7():
    tform = sitk.ReadTransform(os.path.join('data', 'A1.tfm'))
    print_transform_statistics(tform)
# ex7()

def ex8():
    for gamma in np.linspace(start=60,stop=60*4, num=4):
        gamma = int(gamma)
        print(f'Now performing registration with image rotated with gamma = {gamma}')
        rot_matrix = rotation_matrix(pitch=gamma, roll=0,yaw=0)
        A_transform = construct_affine_matrix(rot_matrix)

        # Resample initial image
        interp_ImgT1 = sitk.Resample(vol_sitk, A_transform)
        sitk.WriteImage(interp_ImgT1, fileName=os.path.join('data', f'ImgT1_{str(gamma)}.nii'))    

        imshow_orthogonal_view(interp_ImgT1, title=f'Transformed image with gamma = {str(gamma)}')
# ex8()

def ex9(step_size):
    
    images = []
    ref_img = sitk.ReadImage(os.path.join('data', f'ImgT1_120.nii'))
    for gamma in np.linspace(start=60,stop=60*4, num=4):
        gamma = int(gamma)
        if gamma == 120:
            continue

        print(f'Now performing registration with image rotated with gamma = {gamma}')

        ImgT1_gamma = sitk.ReadImage(os.path.join('data', f'ImgT1_{str(gamma)}.nii'))

        # Perform registration
        img_name = f'ImgT1_{str( gamma )}_interp.nii'
        img_path = os.path.join('data',img_name)
        images.append((gamma,img_path))
        perform_registration(ImgT1_gamma, ref_img,step_size=step_size, write=True, name=img_name, plot=False)

    for gamma,img_path in images:
        interp = sitk.ReadImage(img_path)
        overlay_slices(ref_img, interp, title= f'ImgT1 (red) vs. realigned image with gamma={gamma} (green)')
step_size=20 # 20 Found to work well as optimization step size
# ex9(step_size=step_size)
    # Tuning both pyramid downscaling and step_sizes, quite good average errors were managed to be acquired
        # On individual bases, performances were hamstrung due to large step size, making
        # some images not able to converge. When evaluating the overall performance however,
        # the improvement is grand (240 cannot be aligned correctly with e.g. step_size=10)


# Exercise 10
def ex10():
    fixed_img = sitk.ReadImage(os.path.join('data', f'ImgT1_240.nii'))
    moving_img = vol_sitk
    perform_registration(moving_img, fixed_img, with_composite=True, write=False, name='Img_composite.nii')
# ex10()
    # Exercise is wrong - shouldn't add the last transform no? Removing this, we get 240*, which I believe makes sense, as
    # tform_60 and tform_180 contribute 60* whilst tform_240 contributes 120, which already accomplishes the transformation
    # that we're after.

def ex11():
    # Add synthetic gaussian noise onto image
    moving_image_noisy = sitk.AdditiveGaussianNoise(vol_sitk, mean=0, standardDeviation=200)
    imshow_orthogonal_view(moving_image_noisy, title='Moving image with noise')
# ex11()

def ex11_2():
    fixed_img = vol_sitk
    moving_img = sitk.ReadImage(os.path.join('data', f'ImgT1_240.nii'))
    std=200
    moving_image_noisy = sitk.AdditiveGaussianNoise(moving_img, mean=0, standardDeviation=std)
    imshow_orthogonal_view(moving_image_noisy, title=f'Moving image with noise (std={std})')
    for step in [10,50,150,200]:
        # perform_registration(moving_image_noisy, fixed_img,step_size=step, with_composite=True, write=False, name='Img_noisy.nii')
        perform_registration(moving_image_noisy, fixed_img,step_size=step, write=False, name='Img_noisy.nii',plot=False)
# ex11_2()
    # In general, what we gather from this exercise is, that MSE is robust to uniform gaussian noise.
    # For all step-sizes, it manages to converge, although its loss is quite significant
    # (which is redundant, as this is from the noise)

def ex12():
    fixed_img = vol_sitk
    moving_img = sitk.ReadImage(os.path.join('data', f'ImgT1_240.nii'))
    std=200
    moving_image_noisy = sitk.AdditiveGaussianNoise(moving_img, mean=0, standardDeviation=std)
    # imshow_orthogonal_view(moving_image_noisy, title=f'Moving image with noise (std={std})')
    perform_registration(moving_image_noisy, fixed_img,step_size=20, write=False, name='Img_noisy.nii', noise_robust=True)
ex12()

