import os

def make_objects_xml_string(detected_objects):
    objects_xml_string = ''
    for obj in detected_objects:
        class_name = obj[1]
        score = obj[2]
        xmin = obj[3]
        ymin = obj[4]
        xmax = obj[5]
        ymax = obj[6]
        objects_xml_string += """<object>
        <name>{}</name>
        <pose>Unspecified</pose>
        <truncated>0</truncated>
        <difficult>0</difficult>
        <score>{:0.3f}</score>
        <bndbox>
            <xmin>{}</xmin>
            <ymin>{}</ymin>
            <xmax>{}</xmax>
            <ymax>{}</ymax>
        </bndbox>
    </object>""".format(class_name, score, xmin, ymin, xmax, ymax)
    return objects_xml_string


def make_annotation_xml_string(folder, filename, image_dim, objects_xml, verified):
    '''
      source = not really that important, but it's hardcoded = security cameras
    '''
    
    base = """<annotation{}>
    <folder>{}</folder>
    <filename>{}</filename>
    <path>{}</path>
    <source>
        <database>security cameras</database>
    </source>
    <size>
        <width>{}</width>
        <height>{}</height>
        <depth>3</depth>
    </size>
    <segmented>0</segmented>
    {}
</annotation>""".format(verified, folder, filename, folder, image_dim[1], image_dim[0], objects_xml)
    return base

def inference_to_xml(image_dir, image_basename, orig_image_dim, detected_objects, annotation_dir):
    '''
    image_dir because we put the path in the XML
    '''

    verified_str = ''' verified="no"'''
    objects_xml_string = make_objects_xml_string(detected_objects)
    annotation_xml_string = make_annotation_xml_string(image_dir, image_basename, orig_image_dim, 
        objects_xml_string, verified_str)

    # write to file
    annotation_basename = os.path.splitext(image_basename)[0] + '.xml'

    with open(os.path.join(annotation_dir, annotation_basename), 'w') as f:
        f.write(annotation_xml_string)
    return annotation_xml_string