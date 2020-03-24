import json 
import argparse
import os
import random


# pip install bs4
# pip install lxml

from bs4 import BeautifulSoup

from os import listdir
from os.path import isfile, join

def write_imageset_to_file(imageset_dir, set_name, image_list):
    file_path = os.path.join(imageset_dir, set_name + ".txt")
    image_count = 0
    with open(file_path, "w") as imageset:
        for image_path in image_list:
            image_id = os.path.splitext(image_path)[0]
            imageset.write(image_id + "\n")
            image_count = image_count + 1
    print ('   imageset: ', file_path, " : ", image_count, " written")
            

#
# DEPRECATED - use filter_unverified, it will return both lists
#
def filter_verified(dir_path, file_list):
    verified_list = []
    not_list = []
    for file_name in file_list:
        with open(os.path.join(dir_path, file_name)) as f:
            soup = BeautifulSoup(f, 'xml')
            v = soup.findAll("annotation", {"verified" : "yes"})    # v - only if verified = yes
            if len(v) == 0:
                not_list.append(os.path.splitext(file_name)[0])   # not used
            else: 
                verified_list.append(os.path.splitext(file_name)[0])
    print ('  verified:', len(verified_list), '  not:', len(not_list))
    return verified_list

#
# NOTE - identical to fltered_verifiect
#      - but returns both lists, verified AND not_verifiec
def group_on_verified(dir_path, file_list):
    '''
    Return list of annotations that are NOT verified
    '''
    verified_list = []
    not_list = []
    for file_name in file_list:
        with open(os.path.join(dir_path, file_name)) as f:
            soup = BeautifulSoup(f, 'xml')
            v = soup.findAll("annotation", {"verified" : "yes"})    # v - only if verified = yes
            if len(v) == 0:
                not_list.append(os.path.splitext(file_name)[0])   # not used
            else: 
                verified_list.append(os.path.splitext(file_name)[0])
    print ('  verified:', len(verified_list), '  not:', len(not_list))
    return verified_list, not_list

# get the annotation file list for given directory - then filter to KEEP only the verified ones
def verif_annotation_list(dir_path):
    file_list = [f for f in listdir(dir_path) if isfile(join(dir_path, f))]  # full annotation file list
    verified_list = filter_verified(dir_path, file_list)  # filter this file list KEEPING only the validated annotations
    return verified_list

def group_annotation_list(dir_path):
    file_list = [f for f in listdir(dir_path) if isfile(join(dir_path, f))]  # full annotation file list
    verified_list, not_verified_list = group_on_verified(dir_path, file_list)  # filter this file list KEEPING only the validated annotations
    return verified_list, not_verified_list

# create the image set
#     
def gen_imageset_list(annotation_dir, training_split_tuple):
    verified_list = verif_annotation_list(annotation_dir)
    verified_list_count = len(verified_list)

    # calculate the splits
    # training_split_tuple exptected to be something like (60,30,10)
    # randomly split
    train_count = int(verified_list_count * training_split_tuple[0]/100)
    val_count = int(verified_list_count * training_split_tuple[1]/100)
    test_count = int(verified_list_count * training_split_tuple[2]/100)

    train_list = random.sample(verified_list, train_count)      # pull random out of of verified
    val_test_list = list(set(verified_list) - set(train_list))  # you must REMOVE them from the pool
    val_list = random.sample(val_test_list, val_count)          # pull random out of the remaining pool
    test_list = list(set(val_test_list) - set(val_list))        # finally, subtract val and you have test

    return train_list,  val_list, test_list


# - - - - M A I N - - - - - -
# warning !! i changed main then never ran it or debugged it

def main(args):
    print ("generate image sets")
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_file', type=str, help='relative filepath for config json', default='gen_imagesets_config.json')
    args = parser.parse_args(args)

    # open the config file
    config_filepath = args.config_file
    with open(config_filepath) as json_file:
        data = json.load(json_file)
        train_pct = data['train_pct']
        val_pct = data['val_pct']
        test_pct = data['test_pct']

        train_list, val_list, test_list = gen_imageset_list(data['annotation_dir_list'], (train_pct, val_pct, test_pct))

        write_imageset_to_file(data['imageset_dir_list'], "train", train_list)
        write_imageset_to_file(data['imageset_dir_list'], "val", val_list)
        write_imageset_to_file(data['imageset_dir_list'], "test", test_list)

if __name__ == '__main__':
    import sys
    main(sys.argv[1:])
