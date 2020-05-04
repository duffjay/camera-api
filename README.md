# camera-api
RESTful API for Amcrest cameras

## Python 3.7
conda install pip # might have an issue with conda

pip install flask  
pip install opencv-python
pip install pillow  
pip install bs4  
pip install lxml  
pip install imutils  
pip install matplotlib   # req'd for the graph conversion
pip install contextlib2  
pip install jupyter  
pip install boto3  

Tensorflow or just the TFLite Interpreter?  Well, you will be using a lot of tf.* utilities  
pip install tensorflow-gpu==1.15  

if 8100 - you have an old CPU - you need special TF build - get it off of jmduff/S3  
tensorflow 1.15 required - or you'll get a graph error (incompatible versions)  
python 3.6  
pip install tensorflow-1.15.2-cp36-cp36m-linux_x86_64.whl   

intall kernel on Jupyter:  
python -m ipykernel install --user --name=security  

## get the TensorFlow utils & model => ~/projects
This will also compile the protobufs  
bash ./install_tf_models.sh

you need the label map:  
cp research/object_detection/data/mscoco_label_map.pbtxt ~/projects/camera-api/model/  

you need a tflite model - easiest place to get that is from s3  
you should have created it using the ssd-dag/UnderstandingTensorRT_ConvertGraph notebook  

## TensorFlow (Frozen Graph) vs TF Lite vs EdgeTPU
Which model type to use?  
### TF Lite
I started with tflite because it's simple.   It does not use a GPU.   TFLite is a stepping stone to EdgeTPU (Coral).    Performance was about 0.1 sec per inference.   Not bad but it adds up when you have 4+ image regions.    the reolink2tflite.py program is highly deprecated and will no longer work - but, it's still there if you want to convert to tflite.   This will show you how to use this model type.

### EdgeTPU & Coral
This is a sensible way to go - but I did not.   You convert the *.tflite model to an EdgeTPU model and then it will use the Coral.   The Coral is fast - as fast as the GPU or faster.   But, I did not use this.   The interface to this model is different.   Probably can figure this out by looking at other projects in the github account.  

### TensorFlow - Frozen Graph
This was the preferred solution because I have GPUs.   Requires TF 1.15   
You may need to run:  
  python test_tf_inference.py  
just to compile things - the security program may be slow starting the first time.

## Generating Images

python security.py app_reolink12_tensorflow.json  

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


## tflite vs tensorflow frozen graph

This was originally developed for tflite (mobilenet).   Advantages:
- EdgeTPU compatible
- lightweight
- I knew what I was doing

Without using the Coral TPU, it was about 0.1 second execution (does not utilize GPU).  

But, I have GPUs.   Going to frozen graph - Advantages:
- easier to migrate to ResNet 50 (larger model) at some point
- GPUs with frozen graph are about 5x faster:  0.02  (probably same as a TPU stick in all fairness)
- probably slightly more accurate

While migrating to frozen graph - I migrated to multi-process (1 process/ camera).   And things got messy and disorganized.


