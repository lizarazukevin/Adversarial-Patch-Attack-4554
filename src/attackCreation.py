import cv2
import sys
import os
import numpy as np
import torch
import matplotlib.pyplot as plt


import json
import scipy
import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow.keras.applications.inception_v3 import InceptionV3

import imagenet_stubs
from imagenet_stubs.imagenet_2012_labels import label_to_name, name_to_label

from art.estimators.classification import TensorFlowV2Classifier
from art.preprocessing.expectation_over_transformation import EoTImageRotationTensorFlow
from art.attacks.evasion import ProjectedGradientDescent

# inspiration was taken from https://github.com/Trusted-AI/adversarial-robustness-toolbox

# important parameters
eps = 10.0 / 255.0 # Attack budget for PGD
eps_step = 5.0 / 255.0 # Step size for PGD
num_steps = 30 # Number of iterations for PGD
y_target = np.array([name_to_label("kite")])  # Target class for attack is "kite"
angle_max = 30 # Rotation angle used for evaluation in degrees
eot_angle = angle_max # Maximum angle for sampling range in EoT rotation, applying range [-eot_angle, eot_angle]
eot_samples = 10 # Number of samples with random rotations in parallel per loss gradient calculation

# constant parameters
nb_classes = 1000 # Number of ImageNet classes
preprocessing = (0.5, 0.5) # Preprocessing with (mean, std) for InceptionV3 from input image range [0, 1]
clip_values=(0.0, 1.0) # Clip values for range [0, 1]
input_shape = (299, 299, 3) # Shape of input images

# # loads the yolov5s model trained on COCO128 (default)
# def getDetectionInfo(img_dir):
#     model = torch.hub.load('ultralytics/yolov5', 'yolov5s')
#     model_results = model(img_dir, size=640)
#     info = model_results.pandas().xyxy[0].to_dict(orient="records")

# formulates an EOT attack on an input image
def main():

    # ensures there's a single directory for image
    if len(sys.argv) != 2:
        return
    
    # records image directory
    img_dir = sys.argv[1]
    img_data = np.array(cv2.imread(img_dir))
    print(img_data.shape)

    # checks if the image directory actually works
    if len(img_data.shape) < 2:
        return

    # saves image
    x = (np.expand_dims(img_data, axis=0) / 255.0).astype(np.float32)
    y = np.array([name_to_label("car")])

    model = torch.hub.load('ultralytics/yolov5', 'yolov5s')
    loss = tf.keras.losses.CategoricalCrossentropy(from_logits=False)

    classifier = TensorFlowV2Classifier(model=model,
                                    nb_classes=nb_classes,
                                    loss_object=loss,
                                    preprocessing=preprocessing,
                                    preprocessing_defences=None,
                                    clip_values=clip_values,
                                    input_shape=input_shape)
    
    attack = ProjectedGradientDescent(estimator=classifier,
                                  eps=eps,
                                  max_iter=num_steps,
                                  eps_step=eps_step,
                                  targeted=True)

    x_adv = attack.generate(x=x, y=y_target)

    print(x_adv)

    # # saved as a dictionary the detection information
    # info = getDetectionInfo(img_dir)

    
if __name__ == "__main__":
    np.random.seed(3333)
    main()