import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
from data.LDA import LDA

in_dir = 'data/'
in_file = 'ex6_ImagData2Load.mat'
data = sio.loadmat(in_dir + in_file)
ImgT1 = data['ImgT1']
ImgT2 = data['ImgT2']
ROI_GM = data['ROI_GM'].astype(bool)
ROI_WM = data['ROI_WM'].astype(bool)
vmin1=200
vmin2=60

def ex1():
    fig, ax = plt.subplots(3,2)
    ax[0,0].imshow(ImgT1)
    ax[0,1].imshow(ImgT2)
    T1_nobg = (ImgT1 > vmin1).ravel()
    T2_nobg = (ImgT2 > vmin2).ravel()
    ax[1,0].hist(ImgT1.ravel(), bins=256, range=(vmin1, ImgT1.max()))
    ax[1,1].hist(ImgT2.ravel(), bins=256,range=(vmin2, ImgT1.max()))
    ax[2,0].hist2d(ImgT1.ravel(), ImgT2.ravel(),range=([vmin1, ImgT1.max()], [vmin2, ImgT2.max()]), bins=100, cmap='magma')
    ax[2,1].scatter(ImgT1, ImgT2)
    plt.show()
ex1()
    # Intensity threshold that can roughly separate WM and GM:
        # T1 = 520
        # T2 = 175 # Opposite intensities
    # Can the GM and WM intensity classes be observed in the 2D histogram and scatter plot?
        # Yes, there seem to be two clear clusterings

def ex2():
    _, ax = plt.subplots(1,1)
    ax.imshow(ImgT1, cmap='magma')
    ax.imshow(ROI_GM,alpha=0.5)
    ax.imshow(ROI_WM,alpha=0.5)
    plt.show()
        # Q3: Does the ROI drawings look like what you expect from an expert?
            # I'd say no, as they seem very lacking in the amount of annotation.
ex2()

# Extract ROI from the image slices:
GM1 = ImgT1[ROI_GM > 0]
WM1 = ImgT1[ROI_WM > 0]
GM2 = ImgT2[ROI_GM > 0]
WM2 = ImgT2[ROI_WM > 0]

def ex3():
    _, ax = plt.subplots(1,2)
    ax[0].hist(ImgT1.ravel(), bins=256, range=[vmin1,ImgT1.max()], color='red', alpha=0.5)
    ax[0].hist(GM1.ravel(), bins=256, range=[vmin1,ImgT1.max()], color = 'green')
    ax[0].hist(WM1.ravel(), bins=256, range=[vmin1,ImgT1.max()], color = 'blue')
    ax[1].hist(ImgT2.ravel(), bins=256, range=[vmin2,ImgT1.max()], color='red',alpha=0.5)
    ax[1].hist(GM2.ravel(), bins=256, range=[vmin2,ImgT1.max()],color='green')
    ax[1].hist(WM2.ravel(), bins=256, range=[vmin2,ImgT1.max()],color='blue')
    plt.show()
        # Looking at the histograms, the annotations are as expected wrt. class intensities.
ex3()

# Define feature matrix and target class identifier
X1 = np.c_[GM1,GM2]
X2 = np.c_[WM1,WM2]
X = np.r_[X1,X2]

n_samples, n_features = X.shape[0], X.shape[1]
y = np.zeros(X.shape[0])
y[X1.shape[0]:] = 1



def ex5():
    C1 = X[:X1.shape[0]]
    C2 = X[X1.shape[0]:]
    plt.scatter(C1[:,0], C1[:,1], color='green', marker='x')
    plt.scatter(C2[:,0], C2[:,1], color='black', marker='x')
    plt.show()
        # Clear class separation in scatter plot.
ex5()

# Train weight-vector coeffient for Fisher's LDC:
W = LDA(X, y)

# Remove background
foreground = ImgT1 > vmin1

Xall= np.c_[ImgT1[foreground].ravel(), ImgT2[foreground].ravel()]
Y = np.c_[np.ones((len(Xall), 1)), Xall] @ W.T

# Perform multi-modal classification
PosteriorProb = np.clip(np.exp(Y) / np.sum(np.exp(Y),1)[:, np.newaxis], 0, 1)

# Get indices that are predicted to be in each class
# Ex 9
WM_mask = PosteriorProb[:,0] > 0.5
GM_mask = PosteriorProb[:,1] > 0.5

C1_gt = X[:X1.shape[0]]
C2_gt = X[X1.shape[0]:]

# Show scatterplot of predicted and GT pixels:
def ex10():
    plt.scatter(Xall[WM_mask,0], Xall[WM_mask,1], alpha=0.5,color='blue', marker='x')
    plt.scatter(Xall[GM_mask,0], Xall[GM_mask,1], alpha=0.5,color='red', marker='o')
    plt.scatter(C1_gt[:,0], C1_gt[:,1], color='green', marker='x')
    plt.scatter(C2_gt[:,0], C2_gt[:,1], color='black', marker='o')
    plt.show()
ex10()
    # Decision boundary is clearly set, somewhat wrongfully however.
    
    # Non linear hyperplane would probably perform a little better, however, in the
    # current setting the hyperplane just needs a slight translation. Probably suggests,
    # that there's a lack of sample data at the boundary.

    # Thresholding wouldn't perform as well. This would require that the hyprplane was
    # orthogonal to the classes, which is not the case.

    # The tissues are quite well segmented, will slight inaccuracies. Excluding noise
    # (i.e. background), then the classes should be quite well separated.

# DESCRIPTIONS COULD BE BETTER TO MANY OF THESE EXERCISES! 

def ex11():
    postC1 = np.zeros(ImgT1.shape)
    postC2 = np.zeros(ImgT1.shape)
    postC1[foreground] = WM_mask
    postC2[foreground] = GM_mask    
    _,ax = plt.subplots(ncols=2)
    ax[0].imshow(postC1)
    ax[1].imshow(postC2)
    plt.show()
ex11()
