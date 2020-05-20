import logging

from inference import RegionDetection
from inference import ModelInference
from garage_status import GarageStatus

log = logging.getLogger(__name__)

# important assumptions

cam_garage_outdoor = 0  # ~ 1 sec camera
cam_garage_indoor = 2 
cam_front_porch = 1
cam_back_porch = 3
cam_back_yard = 4
cam_side_yard = 5



# singleton - Status
# - we want only one object/instance of this class
# TODO
# - faces recognized

class Status:
    __instance = None
    def __new__(cls, timestamp):
        if Status.__instance is None:
            print ("new object")
            Status.__instance = object.__new__(cls)
        else:
            print ("reused object")
        Status.__instance.timestamp = timestamp
        Status.__instance.garage_status = GarageStatus("unk", "unk", "unk", "unk")
        return Status.__instance

    def __str__(self):
        status_string = f'Status Object- time: {self.timestamp}\n{self.garage_status}'
        return status_string

    def update_from_detection(self, detection):

        # garaage indoor
        if detection.camera_id == 2:
            # 
            self.garageStatus = self.garage_status.update_from_detection(detection)

        return self 


