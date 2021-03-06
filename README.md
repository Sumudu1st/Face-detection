# Face-detection
Detect human face with live feeds from Webcam.

You will need,

 PyCharm IDE 2018.2

 OpenCV-python 3.2.0.8

 Python 3.6

<img src="Annotation 2020-06-13 150404.jpg">

##### Code Explanation


`import` can load the modules into the current namespace so that you can access the functions and anything else defined within the module using the module name.

``` python
import cv2
import sys
```
`cv2` OpenCV-Python is a library of Python bindings designed to solve computer vision problems.  
`sys` provides access to some variables used or maintained by the interpreter and to functions that interact strongly with the interpreter.

`haarcascade` is basically a classifier which is used to detect the object for which it has been trained for, from the source.

``` python
cascPath = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascPath)
```
`haarcascade_frontalface_default.xml` is a haar cascade designed by OpenCV to detect the frontal face.

define a video capture object 

``` python
video_capture = cv2.VideoCapture(0)
```
`0` is for default camera.

capturing frame by frame

``` python
ret, frame = video_capture.read()
```
Covert frames to gray scale
``` python
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
```
Detecting face
``` python
faces = faceCascade.detectMultiScale(
        gray,   
        scaleFactor=1.1,    
        minNeighbors=5,     
        minSize=(30, 30),   
        flags=cv2.CASCADE_SCALE_IMAGE
    )
```

`faces = faceCascade.detectMultiScale`detecting related object  
`gray,` gray scale  
`scaleFactor=1.1,` scale the face  
`minNeighbors=5,` number of objects  
`minSize=(30, 30),` window size  

Drawing the rectangle around the detacted face.

``` python
for (x, y, w, h) in faces:
 cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
 ```
 
 Displaying the results
 ``` python
 cv2.imshow('Video', frame)
```
Adding command to stop running the program
``` python
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
```
waiting to `q` key to pressed to exit.

When everything is done, release the capture
``` python
video_capture.release()
cv2.destroyAllWindows()
```


