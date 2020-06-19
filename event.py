import logging
import numpy as np
import time

import settings
from status import status_meta_index


log = logging.getLogger(__name__)



# TODO
# - add a dict in status to get camera id by name

event_class_dict = { 
    0 :  {"name" : "street activity", "cameras" : [ ], "notification_emails" : []},
    1 :  {"name" : "walk to back door", "cameras" : [ ], "notification_emails" : []},
    2 :  {"name" : "walk from back door", "cameras" : [ ], "notification_emails" : []},
    3 :  {"name" : "walk to front door", "cameras" : [ ], "notification_emails" : []},
    4 :  {"name" : "walk from front door", "cameras" : [ ], "notification_emails" : []},
    5 :  {"name" : "car enters", "cameras" : [ ], "notification_emails" : []},
    6 :  {"name" : "car leaves", "cameras" : [ ], "notification_emails" : []},
    7 :  {"name" : "two whlr enters", "cameras" : [ ], "notification_emails" : []},
    8 :  {"name" : "two whlr leaves", "cameras" : [ ], "notification_emails" : []},
    9 :  {"name" : "person in garage", "cameras" : [ ], "notification_emails" : []},
    10 :  {"name" : "mail delivered", "cameras" : [ ], "notification_emails" : []},
    11 :  {"name" : "package delivered", "cameras" : [ ], "notification_emails" : []},
    12 :  {"name" : "package taken", "cameras" : [ ], "notification_emails" : []},
    13 :  {"name" : "undefined", "cameras" : [ ], "notification_emails" : []},

}




class Event:
    __instance = None
    def __init__(self, event_class, status):
        self.event_class = event_class
        self.cameras = event_class_dict[event_class]["cameras"]
        self.nofication_emails = event_class_dict[event_class]["notification_emails"]



    def __str__(self):
        status_string = f'Event object - class: {self.event_class} {self.cameras} {self.notification_emails}'
        return status_string

    def notify_event(self):
        # - - send images to s3 - -
        # get the image list
        image_file_list = []
        for camera in cameras:
            meta_index = status_meta_index["last_camera_image_timestamp_start"] + camera    # this is the meta index
            image_timestamp = self.status.history[0, meta_index]                            # this is the timestamp value
            meta_index = status_meta_index["color_code_start"] + camera                     # this is the meta index
            is_color = self.status.history[0, meta_index]
            color_code = 'c'
            if is_color == 0:
                color_code = 'g'
            image_file = os.path.join(settings.snapshot_path, f'{image_timestamp}-{camera}-0-{color_code}.jpg')
            image_file_list.append(image_file)

        # issue e-mail via ses
        return
    
    def log_history(self, history_row_nums):
        '''
        formatted log (print)
        '''

        spaces = '                            '
        with settings.safe_print:
            for row_num in history_row_nums:
                row_desc = settings.row_num_dict[row_num]
                row_desc_len = len(row_desc)
                log.info(f'home_status_history[{row_num}] {row_desc} {spaces[0:-row_desc_len]} {self.history[row_num, 0:40].tolist()}')
        return

    def save_event(self, status_path):
        '''
        '''
        # file name
        base_name = '{}'.format(self.history[0,0])   
        status_name = os.path.join(status_path, base_name + '.npy')
        numpy.save(status_name, self.history)
        return status_name
