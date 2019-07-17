
import cv2
from darkflow.net.build import TFNet
import numpy as np
import matplotlib.pyplot as plt
import time
import sys

img_path = sys.argv[1]

option = {
    'model' : 'cfg/tiny-yolo-voc-2c.cfg',
    'load' : 1500,
    'threshold' : 0.1
}

tfnet = TFNet(option)
capture = cv2.VideoCapture('video.mp4')

colors = [tuple(255 * np.random.rand(3)) for i in range(5)]


img = cv2.imread(img_path, cv2.IMREAD_COLOR)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
result = tfnet.return_predict(img)

for i in range(len(result)):
    tl = (result[i]['topleft']['x'], result[i]['topleft']['y'])
    br = (result[i]['bottomright']['x'], result[i]['bottomright']['y'])
    label = result[i]['label']

    img = cv2.rectangle(img, tl, br, (0,255,0), 7)

    img = cv2.putText(img, label, tl, cv2.FONT_HERSHEY_COMPLEX, 1, (0,0,0), 2)

plt.imshow(img)
plt.show()
