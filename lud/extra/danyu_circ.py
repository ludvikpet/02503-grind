import sys
from skimage import io
import numpy as np
from skimage import measure

# Retrieve segmentation slice and threshold by desired T
segmentation_slice = io.imread(sys.argv[1])
T = 0.5
bin_slice = segmentation_slice > T

# Retrieve regionprops (BLOBs)
label_img = measure.label(bin_slice)
region_props = measure.regionprops(label_img)
airway_slice = region_props[0] # Assuming segmented slice only has one true segmentation

# Compute circularity
circ = circ = (2 * np.sqrt(np.pi * airway_slice.area)) / airway_slice.perimeter
print(f"Circularity: {circ}")
