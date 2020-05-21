
import numpy as np

class RegionDetection:
    def __init__(self, timestamp, camera_id, region_id):
        self.timestmp = timestamp
        self.camera_id = camera_id
        self.region_id = region_id
    def __str__(self):
        detection_string = f'RegionDetection Object - time: {self.timestamp} cam#{self.camera_id}  reg#{self.region_id}'
        return detection_string

class ModelInference(RegionDetection):
    def __init__(self, timestamp, camera_id, region_id, class_array, prob_array):
        super().__init__(timestamp, camera_id, region_id)
        self.class_array = class_array
        self.prob_array = prob_array
    def __str__(self):
        inference_string = f'ModelInference Object - prob: {self.class_array.tolist()}'
        return super(ModelInference, self).__str__() + inference_string


inf = ModelInference(1535532, 2, 0, np.asarray([35, 3]), np.asarray([0.33, 0.55]))
print (inf)