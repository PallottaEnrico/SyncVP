#!/bin/bash
mkdir -p checkpoints

wget -O ./checkpoints/ae_rgb_2f_cityscapes.pth "https://uni-bonn.sciebo.de/s/2YzKxsxepQyykqV/download"
wget -O ./checkpoints/ae_depth_2f_cityscapes.pth "https://uni-bonn.sciebo.de/s/CARuWAzQkaPnXf4/download"
wget -O ./checkpoints/ae_rgb_8f_cityscapes.pth "https://uni-bonn.sciebo.de/s/Y7NL7qxF1s17Ih1/download"
wget -O ./checkpoints/ae_depth_8f_cityscapes.pth "https://uni-bonn.sciebo.de/s/KSspUaOFRHpYWhI/download"

wget -O ./checkpoints/syncvp_cityscapes.pth "https://uni-bonn.sciebo.de/s/AVHLOpAe01SaCFY/download"
