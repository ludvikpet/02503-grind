
import numpy as np 

eye_coord_arr = np.array([[9,8],[7,10]])
right_eye_idx = np.argmax(eye_coord_arr[:,0])
print(right_eye_idx)
left_eye_idx = 1-right_eye_idx
print(left_eye_idx)
print(eye_coord_arr[left_eye_idx,:],eye_coord_arr[right_eye_idx,:])


vec_left_to_right = eye_coord_arr[right_eye_idx,:]-eye_coord_arr[left_eye_idx,:]
print(vec_left_to_right)

