import matplotlib.pyplot as plt 
from matplotlib.patches import Rectangle
import numpy as np 


im = plt.imread("data/DTUSigns2.jpg")
fig, ax = plt.subplots(1,2)
ax[0].imshow(im)
ax[0].set_title("All detections")

bboxes = np.array([[498,1566,1091,2094],[1915,1316,2563,1760],[2082,1483,2406,1603]]) #format: upper left corner, lower right corner in x,y

for bbox in bboxes:
    x1, y1, x2, y2 = bbox
    width = x2 - x1
    height = y2 - y1
    rect = Rectangle((x1, y1), width, height, linewidth=2, edgecolor='r', facecolor='none') #expects bounding boxes in (x_lower_left, y_lower_left), width of box, height of box
    ax[0].add_patch(rect)


#step 1: identify bboxes with overlaps:
def bbox_intersection(boxA, boxB):
    """
    Args:
        boxA (iterable): iterable containing bounding box corners
        boxB (iterable): iterable containing bounding box corners
    Returns:
        boolean: whether the bounding boxes overlap. True when overlap is detected
    """
    """
    SOLUTION
    """
    x1A, y1A, x2A, y2A = boxA
    x1B, y1B, x2B, y2B = boxB
    if x1A >= x2B or x2A <= x1B: 
        return False 
    if y1A >= y2B or y2A <= y1B:
        return False
    # If there is no gap in either X or Y, they must overlap
    return True

    """
    EXERCISE
    """
    x1A, y1A, x2A, y2A = boxA
    x1B, y1B, x2B, y2B = boxB
    
    #implement checks between bounding boxes. Return True when the boxes overlap, and else False.
    

###identifying the complete set of bounding boxes - note, simplified implementation, only valid when at max two bboxes can overlap
overlap_set = []
found_overlap_idx = [] #to mitigate that we do "repeat" registrations
for i in range(len(bboxes)):
    found_overlap = False
    for j in range(i+1, len(bboxes)):
        if bbox_intersection(bboxes[i], bboxes[j]):
            overlap_set.append([bboxes[i], bboxes[j]])
            found_overlap = True
            found_overlap_idx.append(j)

    if found_overlap==False and i not in found_overlap_idx: #case: no overlap found for the box in the complete dataset
        overlap_set.append([bboxes[i]])

fused_detections = [] #save all fused bounding boxes here
for i, set_ in enumerate(overlap_set):
    """
    SOLUTION
    """
    if len(set_)==2:
        detections = np.stack((set_[0],set_[1]))
    else: 
        detections = set_[0][None,:]
    
    mean_box = detections.mean(axis=0)
    fused_detections.append(mean_box)

    """
    EXERCISE: mean fuse bounding boxes
    """
    if len(set_)==2:
        detections = np.stack((set_[0],set_[1]))
    else: 
        detections = set_[0][None,:]
    mean_box = ?
    fused_detections.append(mean_box)
  


##show the post-processed bounding boxes 
ax[1].imshow(im)

for bbox in fused_detections:
    x1, y1, x2, y2 = bbox
    width = x2 - x1
    height = y2 - y1
    rect = Rectangle((x1, y1), width, height, linewidth=2, edgecolor='g', facecolor='none')
    ax[1].add_patch(rect)

ax[1].set_title("Mean fused bounding boxes")
plt.show()
