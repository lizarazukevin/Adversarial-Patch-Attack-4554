import cv2
import sys
import os
import numpy as np
import torch
import matplotlib.pyplot as plt

# loading a pretrained model and capturing its data: https://docs.ultralytics.com/tutorials/pytorch-hub/

# loads the yolov5s model trained on COCO128 (default)
def getDetectionInfo(img_dir):
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s')
    model_results = model(img_dir, size=640)
    info = model_results.pandas().xyxy[0].to_dict(orient="records")

    return info

# plots data here
def plotInfo(info):
    conf, name = [], []
    for object in info:
        conf.append(object["confidence"])
        name.append(object["name"])
    
    name_dict, x, y = {}, [], []
    for _ in range(len(name)):
        # records the index of max confidence to obtain confidence and class name
        curr_index = conf.index(max(conf))

        obj_name = name[curr_index]
        
        # aditional name dictionary allows for multiple classes to be plotted
        if obj_name not in name_dict:
            name_dict[obj_name] = 0
        else:
            name_dict[obj_name] += 1
        
        x.append(name[curr_index] + str(name_dict[obj_name]))
        y.append(conf[curr_index])

        # remove values for next max
        conf.pop(curr_index)
        name.pop(curr_index)

    # plotting occurs here
    fig = plt.figure(figsize=(15,10))
    ax = fig.add_axes([0,0,1,1])
    ax.bar(x, y, edgecolor="black")
    plt.yticks(np.arange(0, 1, 0.2))
    plt.title("Object Detector Output", size=15)
    plt.ylabel("Confidence Level", size=15)
    plt.xlabel("Object Class", size=15)
    plt.savefig('../data/results/test.jpg',bbox_inches='tight', dpi=150)


def main():

    # ensures there's a single directory for image
    if len(sys.argv[1]) == 0:
        return
    
    # records image directory
    img_dir = sys.argv[1]
    img_data = np.array(cv2.imread(img_dir))
    print(img_data.shape)

    # checks if the image directory actually works
    if len(img_data.shape) < 2:
        return

    # saved as a dictionary the detection information
    info = getDetectionInfo(img_dir)

    # plots the graph for class vs confidence of objects recognized
    plotInfo(info)
    

if __name__ == "__main__":
    main()