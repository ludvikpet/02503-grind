import matplotlib.pyplot as plt 
from skimage import io 

dst_img = "data/Hand2.jpg"

dst_img = io.imread(dst_img)
fig, ax = plt.subplots(1,1)
ax.imshow(dst_img)
plt.show()
print("showed hand")
