import cv2


# TODO
# limit custom code - checkout utilities
# https://github.com/tensorflow/models/blob/master/research/object_detection/utils/visualization_utils.py

BBOX_COLOR = list(
                (
                    (80,80,80),
                    (120,120,120),
                    (120,120,0),
                    (120,0,0),
                    (0,120,0),
                    (0,0,120),
                    (255,0,0),
                    (0,255,0),
                    (0,0,255),
                    (102,0,204),
                    (255,102,102),
                    (0,0,128)
                )
            )
#
#  standing outside with an umbrella generates 
#  crazy big bbox dimensions - causing an error
#
def validate_bbox(orig_image_height, orig_image_width, xmin, ymin, xmax, ymax):
    # print ("INPUT: {} {}".format(orig_image_height, orig_image_width))
    truncated = 0
    print ("bbox validate - before: {} {} ({} {}), ({}, {})".format(orig_image_height, orig_image_width, xmin, ymin, xmax, ymax), truncated)

    if xmin > orig_image_width:
        xmin = orig_image_width
        truncated = 1

    if ymin > orig_image_height:
        ymin = orig_image_height
        truncated = 1

    if xmax > orig_image_width:
        xmax = orig_image_width
        truncated = 1
    if ymax > orig_image_height:
        ymax = orig_image_height
        truncated = 1

    # if truncated == 1:
    #     print ("   !!! bbox dimensions clipped !!!")
    
    # print ("bbox validate -  after: {} {} ({} {}), ({}, {})".format(orig_image_height, orig_image_width, xmin, ymin, xmax, ymax), truncated)
    return xmin, ymin, xmax, ymax

def inference_to_image( 
        orig_image,
        bbox_array, class_id_array, prob_array, 
        model_input_dim, label_dict, prob_threshold):
        
        '''
        1 image
        multiple objects detected - thus arrays of (prob, class_id, bbox)
          ONLY inference data for probability > THRESHOLD is being pass in here
          Interate through the objects
          display (with bounding box) only images w/ probability > prob_threshold
        '''
        # you need the scale for drawing bounding boxes
        # - we will draw on the ORIGINAL image (e.g. 480x640)
        # - the model input was (300x300 for example)
        # - the inference is normalized for the model input
        # you need to scale back to the original

        orig_image_width = orig_image.shape[1]
        orig_image_height = orig_image.shape[0]
        objects_per_image_detected = 0
        objects_per_image_ignored = 0
        detected_objects = []                               # empty list of of detected object attributes
        
        if prob_array is not None:
            for i in range(prob_array.size):
                bbox_array = bbox_array.reshape(-1,4)   # reshape from (1,x, 4) to (x, 4)
                class_id = int(class_id_array[i]) + 1  

                # -- different between tflite & edgeTPU -- this is just tflite
                # bbox dimensions - note this is not what you think!
                #    [ymin, xmxin, ymax, xmax]
                xmin = int(bbox_array[i][1] * orig_image_width)
                ymin = int(bbox_array[i][0] * orig_image_height)
                xmax = int(bbox_array[i][3] * orig_image_width)
                ymax = int(bbox_array[i][2] * orig_image_height)

                # draw the bbox - get the color from global color list
                # limited colors defined
                bbox_color_id = class_id % 12

                xmin, ymin, xmax, ymax = validate_bbox(orig_image_height, orig_image_width, xmin, ymin, xmax, ymax)
                cv2.rectangle(orig_image, (xmin,ymin), (xmax, ymax), color=BBOX_COLOR[bbox_color_id],thickness=2)
                # write the class name on the box
                cv2.putText(orig_image, "{} - {:.2f}".format(label_dict[class_id], prob_array[i][0]), 
                    (xmin, ymin), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,0))
                # append to detected objects (using this in annotations)
                detected_objects.append((class_id, label_dict[class_id], prob_array[i], xmin, ymin, xmax, ymax))
                objects_per_image_detected = objects_per_image_detected + 1
 

        enlarged_image = cv2.resize(orig_image, (1280,960), interpolation = cv2.INTER_AREA)
        return enlarged_image, (orig_image_height, orig_image_width), detected_objects