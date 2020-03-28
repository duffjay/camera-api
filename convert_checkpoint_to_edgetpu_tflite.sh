#!/bin/bash

# Convert the checkpoint to tflite model
#  a) chechpoint -> frozen graph
#  b) frozen graph -> tflite
#
#  Expects :
#  - to be executed from tasks/
#  - tflite_model directory exists
#  - code/ (tensorflow/models was cloned)
#  - verify the globals below
#
# MODIFIED for the Security (camera-api) project
# 
#
# ref:  https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/exporting_models.md
#

export PROJECT_DIR=$(pwd)
export CODE_DIR="${PROJECT_DIR}/code"
export TFLITE_DIR="${PROJECT_DIR}/tflite_model"
export TENSORFLOW_DIR="${PROJECT_DIR}/tensorflow_model"
export MODEL_RESEARCH="${HOME}/projects/models/research"
export TRAINED_DIR="${PROJECT_DIR}/model"

export PYTHONPATH="${PYTHONPATH}:${MODEL_RESEARCH}/slim"     # you are setting the path so Python will import from the tensorflow repo
export PYTHONPATH="${PYTHONPATH}:${MODEL_RESEARCH}"

echo "***"
echo $PROJECT_DIR
echo $TFLITE_DIR
echo $PYTHONPATH

# don't ask - this came from the Coral tutorial
#           - this is probably related to the shape of the expected input and output tensors
export INPUT_TENSORS='normalized_input_image_tensor'
export OUTPUT_TENSORS='TFLite_Detection_PostProcess,TFLite_Detection_PostProcess:1,TFLite_Detection_PostProcess:2,TFLite_Detection_PostProcess:3'


# Exit script on error.
set -e
# Echo each command, easier for debugging.
set -x

usage() {
  cat << END_OF_USAGE
  Converts TensorFlow checkpoint to EdgeTPU-compatible TFLite file.

  --checkpoint_num  Checkpoint number, by default 0.
  --pipeline_config
  --help            Display this help.
END_OF_USAGE
}

ckpt_number=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --checkpoint_num)
      ckpt_number=$2
      shift 2 ;;
    --pipeline_config)
      pipeline_config=$2
      shift 2 ;; 
    --help)
      usage
      exit 0 ;;
    --*)
      echo "Unknown flag $1"
      usage
      exit 1 ;;
  esac
done


rm ${TENSORFLOW_DIR} -rf
rm ${TFLITE_DIR} -rf

# test before you run -- this gets confusing going between platforms

# is the checkpoint (raw graph) there?
echo "-- check for model checkpoint (the raw graph):", $TRAINED_MODEL_DIR, $ckpt_number
ls $TRAINED_DIR/*$ckpt_number*


# - we don't think labels.txt is used, so don't copy it
# - this code can be deleted - if everything runs :)
# echo "copy labels.txt file from ${DATASET_DIR}"
# cp ${DATASET_DIR}/labels.txt ${TFLITE_DIR}
echo " - - - CKPT ==> tensorflow frozen graph - - -"
python ${MODEL_RESEARCH}/object_detection/export_inference_graph.py \
  --input_type image_tensor \
  --pipeline_config_path="${PROJECT_DIR}/${pipeline_config}" \
  --trained_checkpoint_prefix="${TRAINED_DIR}/model.ckpt-${ckpt_number}" \
  --output_directory="${TENSORFLOW_DIR}"

echo " - - - CKPT ==> tflite frozen graph - - -"
python ${MODEL_RESEARCH}/object_detection/export_tflite_ssd_graph.py \
  --pipeline_config_path="${PROJECT_DIR}/${pipeline_config}" \
  --trained_checkpoint_prefix="${TRAINED_DIR}/model.ckpt-${ckpt_number}" \
  --output_directory="${TFLITE_DIR}" \
  --add_postprocessing_op=true

echo " - - - - - - - -"
echo INPUT_TENORS
echo " - - - tflite frozen graph ==> *.tflite - - - "
tflite_convert \
  --output_file="${TFLITE_DIR}/output_tflite_graph.tflite" \
  --graph_def_file="${TFLITE_DIR}/tflite_graph.pb" \
  --inference_type=QUANTIZED_UINT8 \
  --input_arrays="${INPUT_TENSORS}" \
  --output_arrays="${OUTPUT_TENSORS}" \
  --mean_values=128 \
  --std_dev_values=128 \
  --input_shapes=1,300,300,3 \
  --change_concat_input_ranges=false \
  --allow_nudging_weights_to_use_fast_gemm_kernel=true \
  --allow_custom_ops

echo "TFLite graph generated at ${TFLITE_DIR}/output_tflite_graph.tflite"
