#!/bin/bash 

#
#  this will install the tensorflow/models repo
#  - you need this for the actual model and a bunch of utilities
#  - it will also compmile the protobufs
#

# MUST BE in <top project folder>/tasks
# clone the repo into code/models
echo "--- git clone ---"
cd ${HOME}/projects
git clone https://github.com/tensorflow/models.git

# get the protobuf compiler
# ref: https://developers.google.com/protocol-buffers/docs/pythontutorial
echo "--- get the protobuf compiler ---"
cd models/research
wget -O protobuf.zip https://github.com/google/protobuf/releases/download/v3.0.0/protoc-3.0.0-linux-x86_64.zip
unzip protobuf.zip
# compile the proto
# your are in code/models
echo "--- compile protobufs ---"
./bin/protoc object_detection/protos/*.proto --python_out=.

# clean up
echo "--- clean up ---"
rm protobuf.zip
rm -r bin
rm include -r

echo "--- done! ---"

