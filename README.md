# camera-api
RESTful API for Amcrest cameras

## Python 3.7
pip install flask  
pip install opencv-python
pip install pillow
pip install imutils  

Tensorflow or just the TFLite Interpreter?  Well, you will be using a lot of tf.* utilities  
pip install tensorflow-gpu==1.15  

if 8100 - you have an old CPU - you need special TF build - get it off of jmduff/S3  
pip install ~/Downloads/tensorflow-1.14.1-cp36-cp36m-linux_x86_64.whl  

intall kernel on Jupyter:  
python -m ipykernel install --user --name=camera-api  

## get the TensorFlow utils & model => ~/projects
This will also compile the protobufs  
bash ./install_tf_models.sh

you need the label map:  
cp research/object_detection/data/mscoco_label_map.pbtxt ~/projects/camera-api/model/  

you need a tflite model - easiest place to get that is from s3  
you should have created it using the ssd-dag/UnderstandingTensorRT_ConvertGraph notebook  

## Generating Images

## Labeling Images

using github labelimg  
follow the install directions found in the README.md  
$ conda activate labelimg  
$ cd ~/projects/labelImg  
$ python labelImg.py  

## labeled images -> tfrecords
hint:  tar xvf tarball.tar.gz  --strip-components=1  

put annotations (xml) in annotations/  
put images in jpg_images/  





