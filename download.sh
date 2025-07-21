#!/bin/bash
mkdir -p checkpoints


if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <dataset_name>"
    echo "Available models trained on: cityscapes, bair"
    exit 1
fi

DATASET=$1

if [ "$DATASET" == "cityscapes" ]; then
  wget -O ./checkpoints/ae_rgb_2f_cityscapes.pth "https://uni-bonn.sciebo.de/s/2YzKxsxepQyykqV/download"
  wget -O ./checkpoints/ae_depth_2f_cityscapes.pth "https://uni-bonn.sciebo.de/s/CARuWAzQkaPnXf4/download"
  wget -O ./checkpoints/ae_rgb_8f_cityscapes.pth "https://uni-bonn.sciebo.de/s/Y7NL7qxF1s17Ih1/download"
  wget -O ./checkpoints/ae_depth_8f_cityscapes.pth "https://uni-bonn.sciebo.de/s/KSspUaOFRHpYWhI/download"
  wget -O ./checkpoints/syncvp_cityscapes.pth "https://uni-bonn.sciebo.de/s/AVHLOpAe01SaCFY/download"
elif [ "$DATASET" == "bair" ]; then
  wget -O ./checkpoints/ae_rgb_2f_bair.pth "https://uni-bonn.sciebo.de/s/AjsTZWEeTCz2cpk/download"
  wget -O ./checkpoints/ae_depth_2f_bair.pth "https://uni-bonn.sciebo.de/s/kM6y9zsisoXDtCS/download"
  wget -O ./checkpoints/ae_rgb_8f_bair.pth "https://uni-bonn.sciebo.de/s/r7QPrpatNcjbSoX/download"
  wget -O ./checkpoints/ae_depth_8f_bair.pth "https://uni-bonn.sciebo.de/s/NkBKFMJrA4ii8ec/download"
  wget -O ./checkpoints/syncvp_bair.pth "https://uni-bonn.sciebo.de/s/EXgqj66RSjeprR9/download"
else
  echo "Unknown dataset: $DATASET"
  echo "Available models trained on: cityscapes, bair"
  exit 1
fi