# Adversarial-Patch-Attack-4554
This is the git repo for Adversarial Patch Attacks for the Camouflaging of Pedestrians and Cars From Computer Vision Models in Autonomous Driving.

*SRC* - All the source code used or created

*Images*- All relevant images for our HTML file

*Data* - All the current tests, patch, and resulting images saved

**Running Our Code**

insertPatch.py - this script takes in two arguments, these are the main image and the patch image, once entered the output is saved to results folder in data
EX: python inserPatch.py img1_dir patch1_dir

imgDetection.py - this script saves both the image with bounding boxes and a bar graph incapsulating the confidence intervals and classes, a pre-trained single detector (YOLOv5s) is loaded that's been trained on COCO128, saves to results folder in data
EX: python imgDetection.py img_dir

videoDetection.py - this script is still in development, will take in a single argument and analyze the frequency at which objects are detected.

attackCreation.py - this script intakes a single argument, an image and assigns itself with various parameters to create an adversarial attack