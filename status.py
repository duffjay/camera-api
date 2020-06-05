import logging
import numpy as np
import time

import settings

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

unknown = -1


# status history dict
# cam -many- regions -many- history stacks
# e.g.
# cam 0 -> 6 regions -> car, person, bike histories
# so you need a map(camera, region, class) => numpy axis coordinates

# name
# regions

# MUST be in same order as camera config in the config file
#     e.g.  camera 0 == garage outdoor in both config file and this dict
status_history_dict = { 
    "cam0" :  {"id" : 0, "name" : "garage outdoor", "regions" : [ 
        {"id" : 0, "name" : "full",        "history" : {"car" : 0, "person" : 0, "2whlr": 0}},
        {"id" : 1, "name" : "driveway",    "history" : {"car" : 0, "person" : 0, "2whlr": 0}},
        {"id" : 2, "name" : "backdoor",    "history" : {"car" : 0, "person" : 0, "2whlr": 0}},
        {"id" : 3, "name" : "parking pad", "history" : {"car" : 0, "person" : 0, "2whlr": 0}},
        {"id" : 4, "name" : "close",       "history" : {"car" : 0, "person" : 0, "2whlr": 0}},
        {"id" : 5, "name" : "backyard",    "history" : {"car" : 0, "person" : 0, "2whlr": 0}}
        ] },
    "cam1" : {"id" : 1, "name" : "front door", "regions" : [ 
        {"id" : 0, "name" : "full",        "history" : {"car" : 0, "person" : 0, "2whlr": 0}},
        {"id" : 1, "name" : "driveway 0",  "history" : {"car" : 0, "person" : 0, "2whlr": 0}},
        {"id" : 2, "name" : "driveway 1",  "history" : {"car" : 0, "person" : 0, "2whlr": 0}},
        {"id" : 3, "name" : "driveway 2",  "history" : {"car" : 0, "person" : 0, "2whlr": 0}},
        {"id" : 4, "name" : "driveway 3",  "history" : {"car" : 0, "person" : 0, "2whlr": 0}},
        {"id" : 5, "name" : "stair 0",     "history" : {"car" : 0, "person" : 0, "2whlr": 0}},
        {"id" : 6, "name" : "stair 1",     "history" : {"person" : 0, "package": 0}},
        {"id" : 7, "name" : "stair 2",     "history" : {"person" : 0, "package": 0}}
        ] },
    "cam2" : {"id" : 2, "name" : "garage indoor", "regions" : [ 
        {"id" : 0, "name" : "full",        "history" : {"gmark_door" : 0, "gmark_car" : 0, "car" : 0, "person" : 0, "2whlr": 0}},
        {"id" : 1, "name" : "left door",    "history" : {"gmark_door" : 0, "gmark_car" : 0, "car" : 0, "person" : 0, "2whlr": 0}},
        {"id" : 2, "name" : "right door",    "history" : {"car" : 0, "person" : 0, "2whlr": 0}},
        {"id" : 3, "name" : "window", "history" : {"car" : 0, "person" : 0, "2whlr": 0}}
        ] }
}

history_depth = 120     # 1/2 second increments, depth of stack



def configure_history_map(map_dict):
    '''
    the status history dict is UNCONFIGURED -- the values are all 0
       why?  because it's easier to automate the row assignment - especially as you keep changing this
    
    this function will simply assign a row number sequentially
    row 0 == meta data (e.g. timestamp of the update)
    rows 1:  == the history

    to facilitate a reverse lookup, this function will also create a row number map
      key:  row number
      value:  camera/region/catagory   e.g. 0:1:person
    '''

    row_num_dict = {}
    row_num_dict[0] = "meta data"
    # row = 0 == meta data like the date stamp
    # history will start with row = 1
    np_row = 1
    # for each camera
    for camera_id, camera_short_name in enumerate(map_dict.keys()):         # cam0, cam1, cam2 etc
        # print (f'camera: {camera_id} {camera_short_name}')
        camera_id = map_dict[camera_short_name]["id"]                     # map_dict['cam0']
        camera_name = map_dict[camera_short_name]["name"]
        camera_regions = map_dict[camera_short_name]["regions"]           # map_dict['cam0']['regions'] => list

        for region_id, region in enumerate(camera_regions):
            # print (f'region: {region["name"]}')
            # history stacks
            history_stacks = region["history"]
            for stack_id, stack in enumerate(history_stacks.keys()):
                # print (f'history: {stack_id} {stack}')
                map_dict[camera_short_name]["regions"][region_id]["history"][stack] = np_row
                # f'config history map: {camera_short_name} {map_dict[camera_short_name]["regions"][region_id]["name"]
                # f' {map_dict[camera_short_name]["regions"]["history"][stack]}'
                # f' => {map_dict[camera_short_name]["regions"][region_id]["history"][stack]}')
                log.info (
                    f'config history map: {camera_short_name}-{region_id}:{map_dict[camera_short_name]["regions"][region_id]["name"]}--{stack} == '
                    f' {map_dict[camera_short_name]["regions"][region_id]["history"][stack]}'
                )
                row_num_dict[np_row] = f'{camera_short_name}-{region_id}:{map_dict[camera_short_name]["regions"][region_id]["name"]}--{stack}'
                np_row = np_row + 1
        np_row_count = np_row

    return map_dict, np_row_count, row_num_dict

def get_history_np_row(map_dict, camera_id, region_id, history_catagory):
    '''
    map_dict must be configured 
    '''
    camera_short_name = f'cam{camera_id}'
    region = map_dict[camera_short_name]["regions"][region_id]
    np_row = region["history"][history_catagory]
    return np_row




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
        Status.__instance.garage_status = GarageStatus(unknown, unknown, unknown, unknown)
        # create the history numpy array
        Status.__instance.history = np.full([settings.history_row_count, history_depth], -1, dtype=int)
        # set the timestamp
        Status.__instance.history[0,0] = int(time.time() * 10)

        # TODO import numpy as np, shape history
        return Status.__instance

    def __str__(self):
        status_string = f'Status Object- time: {self.timestamp}\n{self.garage_status}'
        return status_string

    def update_from_detection(self, detection):
        '''
        called by image_consumer (per camera:region)
        '''

        # garage indoor
        if detection.camera_id in (cam_garage_outdoor, cam_garage_indoor):
            # 
            self.garageStatus = self.garage_status.update_from_detection(self, detection)

        return self 

    def update_history(self, image_time, history_row_num, status_code, comment=''):
        '''
        update the correct row of the history matrix
        '''
        
        # remember - timestamp == time * 10 == 1/10s of seconds
        hist_timestamp = self.history[0,0]                               # history array timestamp
        det_timestamp = image_time                                  # timestamp when image was grabbed
        index = int((hist_timestamp - det_timestamp) / 10)    # add  1 because position 0 == timestamp
        
        #
        log.info(f'Status.update_history: {self.history[history_row_num, :10].tolist()}'
            f'  image_time:{image_time}  status:{status_code}  index: {index} {comment}')
        if index < (history_depth) and index > 0:
            self.history[history_row_num, index] = status_code
        return self.history
    
    def log_history(self, history_row_nums):
        '''
        formatted log (print)
        '''

        spaces = '                            '
        for row_num in history_row_nums:
            row_desc = settings.row_num_dict[row_num]
            row_desc_len = len(row_desc)
            log.info(f'home_status_history[{row_num}] {row_desc} {spaces[0:-row_desc_len]} {self.history[row_num, 0:25].tolist()}')
        return


