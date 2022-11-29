import cv2
import sys
import os
import numpy as np
from PIL import Image
from datetime import datetime

# pastes the image on the upper left area
def main():

    # ensures there's a single directory for image
    if len(sys.argv) != 3:
        return
    
    # records image directory
    img_dir = sys.argv[1]
    patch_dir = sys.argv[2]
    img_data = np.array(cv2.imread(img_dir))
    patch_data = np.array(cv2.imread(patch_dir))

    # checks if the image directory actually works
    if len(img_data.shape) < 2 or len(patch_data) < 2:
        return

    # opens the images
    img = Image.open(img_dir)
    patch = Image.open(patch_dir)

    # combines both images such that patch is at the top left area and saves them
    now = datetime.now()
    time = now.strftime("%H%M%S")
    out_dir = '../data/results/combined'+ time +'.jpg'
    img.paste(patch, (0,0))
    img.save(out_dir)

    print("Combined both images!")

if __name__ == "__main__":
    main()