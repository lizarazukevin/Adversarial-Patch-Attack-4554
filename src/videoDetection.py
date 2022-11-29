import yolov5
import cv2
import numpy as np
from yolov5 import detect
import matplotlib
import sys
import os

# inspo from: https://docs.opencv.org/4.x/dd/d43/tutorial_py_video_display.html

def main():
    # checks the if the directory is valid for the image to be analyzed
    if len(sys.argv[1]) > 0:
        media_path = sys.argv[1]
        vid_read = cv2.VideoCapture(media_path)

        while vid_read.isOpened():
            ret, frame = vid_read.read()
            # if frame is read correctly ret is True
            if not ret:
                print("Can't receive frame (stream end?). Exiting ...")
                break
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            cv2.imshow('frame', gray)
            if cv2.waitKey(1) == ord('q'):
                break
    
    vid_read.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()