import cv2 as cv
from pathlib import Path
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from mtcnn.mtcnn import MTCNN
import matplotlib
matplotlib.use('TkAgg') 

#directory setting
curr_dir=Path.cwd()
img_dir=curr_dir / 'Testingimage' / 'f.jpg'
print(img_dir)
train_dir='E:/498R/dataset_simple/Train'
test_dir='E:/498R/dataset_simple/Test'
validation_dir='E:/498R/dataset_simple/Validation'

##Image Loading
test_img=cv.imread(str(img_dir))
cv.imshow('Image',test_img)
cv.waitKey(50)

#detect Faces
#before Detection convert The imag To RGB to feed in MTCNN
bgr_img=cv.cvtColor(test_img,cv.COLOR_BGR2RGB)
detector=MTCNN()
img_result=detector.detect_faces(bgr_img)
x,y,w,h=img_result[0]['box']
keypoints=img_result[0]['keypoints']
print(keypoints)
cv.imshow("Rimage",test_img)

# #working with Result
img=cv.rectangle(test_img,(x,y),(x+w,y+h),(0, 255, 255),(2))
for k in keypoints.values(): 
   img=cv.circle(img,k,4, (0, 255, 255), 4)
image_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)
plt.imshow(image_rgb)
cv.imshow("Image",img)
# Display the plot
plt.show()
cv.waitKey(10)


#image resizing so that only face pixel can be feed to encoder
cv.imshow("Test_Image",test_img)
cv.waitKey(10)

resize_img=cv.resize(image_rgb[y:y+h,x:x+w],(160,160))
plt.imshow(resize_img)
plt.show()
cv.destroyAllWindows()