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
    return verified_list, not_list


def group_annotation_list(dir_path):
    '''
    for given path - return annotations grouped verified/unverified
    '''
    file_list = []
    for f in listdir(dir_path):
        full_file_path = os.path.join(dir_path, f)
        if isfile(join(dir_path, f)):
            file_list.append(full_file_path)

    verified_list, not_verified_list = group_on_verified(dir_path, file_list)  # filter this file list KEEPING only the validated annotations
    return verified_list, not_verified_list

def print_audit_list(image_list, audit_count):
    print ('\n')
    for i in range(audit_count):
        print ("        ", image_list[i])
    return

# create the image set
#     
def gen_imageset_list(annotation_root, training_split_tuple):
    '''
    generate three (3) image set lits (train, val, test)
    from the sum of directories in the annotation dir list
    '''
    print ("-- MAKING IMAGES LISTS - train/val/test")
    # top level (root) path given, get all directories
    root_contents = os.listdir(annotation_root)
    annotation_subdirectories = []
    for entry in root_contents:
        full_entry_path = os.path.join(annotation_root, entry)
        if os.path.isdir(full_entry_path):
            annotation_subdirectories.append(full_entry_path)
    print ("   found {} subdirectories".format(len(annotation_subdirectories)))
    # now you have all subdirectories == annotation_subdirectories

    total_verified_list = []  # empty list
    # loop thru each annotation dir in the list
    for i, annotation_dir in enumerate(annotation_subdirectories):
        single_verified_list, single_nonverified_list  = group_annotation_list(annotation_dir)
        single_verified_count = len(single_verified_list)
        single_nonverified_count = len(single_nonverified_list)
        total_verified_list.extend(single_verified_list)
        print ("   making list of verified annotations -- verified count {} / non {} in {}".format(
            single_verified_count, single_nonverified_count, annotation_dir))
    
    verified_list_count = len(total_verified_list)
    print ("-- TOTAL list of verified annotations -- verfied count {}".format(verified_list_count))

    # calculate the splits
    # training_split_tuple exptected to be something like (60,30,10)
    # randomly split
    train_count = int(verified_list_count * training_split_tuple[0]/100)
    val_count = int(verified_list_count * training_split_tuple[1]/100)
    test_count = int(verified_list_count * training_split_tuple[2]/100)
    print ("   split counts: {} {} {}".format(train_count, val_count, test_count))

    print ("   making randomized lists")
    train_list = random.sample(total_verified_list, train_count)      # pull random out of of verified
    val_test_list = list(set(total_verified_list) - set(train_list))  # you must REMOVE them from the pool
    val_list = random.sample(val_test_list, val_count)          # pull random out of the remaining pool
    test_list = list(set(val_test_list) - set(val_list))        # finally, subtract val and you have test
    # QC - just print a few to make sure you have full path & you have random list
    print_audit_list(train_list, 10)
    print_audit_list(val_list, 10)
    print_audit_list(test_list, 10)

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
