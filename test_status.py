import inference
import status
import numpy as np

import time
import logging

current_timestamp = time.time()
home_status = status.Status(current_timestamp)
print (f'{status}\n\n' )

# create an inference
class_array = np.asarray([3])
prob_array  = np.asarray([0.7207128])
bbox_array  = np.asarray([[0.5771041, 0.0, 0.9947256, 0.24670777]])

inf = inference.ModelInference(class_array, prob_array, bbox_array)


# create a detection
camera_id = 2
region_id = 3
new_objects = 1
dup_objects = 0
det = inference.RegionDetection(camera_id, region_id, new_objects, dup_objects, inf)

print (f'DETECTION:\n {det}')

# update the status  with the detection
home_status.update_from_detection(det)
print (f'{status}\n\n' )