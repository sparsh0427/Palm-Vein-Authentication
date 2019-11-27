# Palm-Vein-Authentication-System

#### Every person has a unique vein pattern in their palms. Palm vein authentication is the process of using this pattern to identify who you are. Despite being not very well known, it has potential to become one of the best forms of biometric authentication. It's contactless, extremely reliable, and difficult to impersonate. Here's how to make a simple but highly accurate one using a Raspberry Pi, OpenCV, and TensorFlow.

### Getting the Image
#### The first step is to get an image.. but how can we get an accurate picture of our veins? Turns out, hemoglobin in our blood absorbs infrared light. So, if we take some infrared LEDs (mine were 940nm) and position them under one's hand, we should see veins!! Let's create a setup where we can get a consistent image every time.

#### An ordinary shoebox with a palm-sized hole cut out above the camera worked perfectly for me. 

![Setup](C:\Users\Sparsh\Desktop\qq.jpg)

The circuitry is very simple - we just need to power the IR LEDs. I used 5 LEDs connected in series with a 100 ohm resistor (you may need a different resistance depending on your LEDs) and 9V battery. The RaspberryPi is on top of the breadboard, with the camera resting on the battery facing up.

##### Let's set up auto-cropping, as we're only concerned with the palm. This is the command I used to produce a 600x600 image (you'll want it to be square).

``` raspistill -vf -w 600 -h 600 -roi 0.46,0.34,0.25,0.25 -o pic.jpg ```

##### Now we have this cropped image of our palm. We need to perform some image processing before we can actually make use of it.

### Image Processing

##### You'll need OpenCV installed on your Pi. First, we load the image and convert it to grayscale.

##### Load the 600x600 image and convert to grayscale
``` img = cv2.imread("pic.jpg") ```
``` gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)```

##### Let's start by reducing some of the noise. Luckily, OpenCV has a function for that:

``` noiseReduced = cv2.fastNlMeansDenoising(gray) ```

Much smoother. Now, we need to increase the contrast to really make the veins stand out. The method I used was histogram equalization. This distributes the intensities of the pixels in the image, "equalizing" the histogram. We then invert the image, since many OpenCV functions assume the background is black and foreground is white.

``` 
# equalize hist
kernel = np.ones((7,7),np.uint8)
img = cv2.morphologyEx(noiseReduced, cv2.MORPH_OPEN, kernel)
img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
img_yuv[:,:,0] = cv2.equalizeHist(img_yuv[:,:,0])
img_output = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)
```

That made quite a big difference. A lot of the "skin" is gone (now black), with the vein pattern being largely white. It's still not quite ready yet - there's a lot of redundant data in this image.

Similarly, we want to do this with the vein image. Let's "skeletonize" it, using repeated erosion.

```
img = gray.copy()
skel = img.copy()
skel[:,:] = 0
kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (5,5))
while cv2.countNonZero(img) > 0:
    eroded = cv2.morphologyEx(img, cv2.MORPH_ERODE, kernel)
    temp = cv2.morphologyEx(eroded, cv2.MORPH_DILATE, kernel)
    temp  = cv2.subtract(img, temp)
    skel = cv2.bitwise_or(skel, temp)
    img[:,:] = eroded[:,:]

```
I applied a quick threshold to make the veins more visible. Every pixel which is 5 or higher (everything very dark gray or lighter) will become 255 (white).

```
ret, thr = cv2.threshold(skel, 5,255, cv2.THRESH_BINARY);
```

To see how accurate this was, I overlayed the vein pattern over the original image to see if there was a correlation.

It's looking good! Not perfect, but it should be more than good enough for our purposes.

### Authnetication

To be able to authenticate, you'll need TensorFlow installed. We'll be using a basic classification method.
