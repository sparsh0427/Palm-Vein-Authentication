# Palm-Vein-Authentication-System

#### Every person has a unique vein pattern in their palms. Palm vein authentication is the process of using this pattern to identify who you are. Despite being not very well known, it has potential to become one of the best forms of biometric authentication. It's contactless, extremely reliable, and difficult to impersonate. Here's how to make a simple but highly accurate one using a Raspberry Pi, OpenCV, and TensorFlow.

### Getting the Image
#### The first step is to get an image.. but how can we get an accurate picture of our veins? Turns out, hemoglobin in our blood absorbs infrared light. So, if we take some infrared LEDs (mine were 940nm) and position them under one's hand, we should see veins!! Let's create a setup where we can get a consistent image every time.

#### An ordinary shoebox with a palm-sized hole cut out above the camera worked perfectly for me. 

![Setup](C:\Users\Sparsh\Desktop\qq.jpg)
