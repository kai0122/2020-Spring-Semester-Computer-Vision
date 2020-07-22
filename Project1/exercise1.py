# Po-Kai Yang
# Computer Vision: Exercise 1
# python version: python2.7

import numpy as np
import cv2
import os
from absl import app
from skimage import img_as_ubyte
import time
from tqdm import tqdm



def readVideo(path):
    cap = cv2.VideoCapture(path)
    return cap


def getVideoWriter(videoIn, fps):
    frame_width = int(videoIn.get(3))
    frame_height = int(videoIn.get(4))

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('./temporaryGeneratedFiles/outpy.avi',cv2.VideoWriter_fourcc('M','J','P','G'), fps, (frame_width,frame_height))
    return out


def switchGrayBGR(videoIn, videoOut, fps):
    # 4 second
    Index_FirstEffect = 5 * fps
    currentColor = "color"
    flagNumberFirst = [55,85,232,262]

    countingIndex = 0
    pbar = tqdm(total=100)
    while(countingIndex < Index_FirstEffect):
        # Capture frame-by-frame
        ret, frame = videoIn.read()

        # if frame is read correctly ret is True
        if not ret:
            print("Stream End...")
            break
        if countingIndex in flagNumberFirst:
            if currentColor == "color":
                # change to gray
                currentColor = "gray"
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                backtorgb = cv2.cvtColor(gray,cv2.COLOR_GRAY2RGB)
                videoOut.write(backtorgb)
            else:
                # change to color
                currentColor = "color"
                videoOut.write(frame)
        else:
            if currentColor == "color":
                videoOut.write(frame)
            else:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                backtorgb = cv2.cvtColor(gray,cv2.COLOR_GRAY2RGB)
                videoOut.write(backtorgb)
        countingIndex += 1
        if ((countingIndex/Index_FirstEffect)*100) >= 100:
            pbar.update(100-pbar.n)
        else:
            pbar.update((((countingIndex/Index_FirstEffect)*100)-pbar.n))
    pbar.close()


def blurBilateralGaussian(videoIn, videoOut, fps):
    # 8 second
    Index_SecondEffect = 7.75 * fps
    flagNumberSecond = [1.75*fps,2.75*fps,4*fps,5*fps,6*fps,7*fps]
    secondPartVariable = 5

    countingIndex = 0
    pbar = tqdm(total=100)
    while(countingIndex < Index_SecondEffect):
        # Capture frame-by-frame
        ret, frame = videoIn.read()

        # if frame is read correctly ret is True
        if not ret:
            print("Stream End...")
            break

        if countingIndex <= flagNumberSecond[0]:
            videoOut.write(frame)
        elif countingIndex <= flagNumberSecond[1]:
            blur = cv2.bilateralFilter(frame, 40, secondPartVariable, secondPartVariable)
            secondPartVariable += 5
            videoOut.write(blur)
        elif countingIndex <= flagNumberSecond[2]:
            blur = cv2.bilateralFilter(frame, 40, secondPartVariable, secondPartVariable)
            secondPartVariable -= 5
            videoOut.write(blur)
        elif countingIndex <= flagNumberSecond[3]:
            videoOut.write(frame)
            secondPartVariable = 15
        elif countingIndex <= flagNumberSecond[4]:
            blur = cv2.GaussianBlur(frame, (secondPartVariable, secondPartVariable), 0)
            secondPartVariable += 2
            videoOut.write(blur)
        elif countingIndex <= flagNumberSecond[5]:
            blur = cv2.GaussianBlur(frame, (secondPartVariable, secondPartVariable), 0)
            secondPartVariable -= 2
            videoOut.write(blur)
        else:
            videoOut.write(frame)
        countingIndex += 1
        if ((countingIndex/Index_SecondEffect)*100) >= 100:
            pbar.update(100-pbar.n)
        else:
            pbar.update((((countingIndex/Index_SecondEffect)*100)-pbar.n))
    pbar.close()


def detectObjectRGBHSV(videoIn, videoOut, fps):
    # 8 second
    Index_ThirdEffect = 7 * fps
    flagNumberThird = [2.5*fps, 4.75*fps]

    countingIndex = 0
    pbar = tqdm(total=100)
    while(countingIndex < Index_ThirdEffect):
        # Capture frame-by-frame
        ret, frame = videoIn.read()

        # if frame is read correctly ret is True
        if not ret:
            print("Stream End...")
            break

        if countingIndex < flagNumberThird[0]:
            # Convert BGR to HSV
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

            # lower mask (0-10)
            lower_red = np.array([0,50,50])
            upper_red = np.array([3,255,255])
            mask0 = cv2.inRange(hsv, lower_red, upper_red)

            # upper mask (170-180)
            lower_red = np.array([177,50,50])
            upper_red = np.array([180,255,255])
            mask1 = cv2.inRange(hsv, lower_red, upper_red)

            # join my masks
            mask = mask0+mask1
            null = frame
            null[null >= 0] = 255
            output = cv2.bitwise_and(null, null, mask = mask)
            videoOut.write(output)
        else:
            # create NumPy arrays from the boundaries
            lower = np.array([0, 0, 100], dtype = "uint8")
            upper = np.array([50, 56, 255], dtype = "uint8")
            # find the colors within the specified boundaries and apply
            # the mask
            mask = cv2.inRange(frame, lower, upper)
            null = frame
            null[null >= 0] = 255
            output = cv2.bitwise_and(null, null, mask = mask)

            if countingIndex < flagNumberThird[1]:
                videoOut.write(output)
            else:
                kernel = cv2.getStructuringElement(cv2.MORPH_CROSS,(7,7))
                opening = cv2.morphologyEx(output, cv2.MORPH_OPEN, kernel)
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(7,7))
                closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)
                # show the images
                videoOut.write(closing)
        countingIndex += 1
        if ((countingIndex/Index_ThirdEffect)*100) >= 100:
            pbar.update(100-pbar.n)
        else:
            pbar.update((((countingIndex/Index_ThirdEffect)*100)-pbar.n))
    pbar.close()


def sobelEdgeDetection(videoIn, videoOut, fps):
    # 5 second
    Index_Effect = 5.9 * fps
    flagNumber = [2.75*fps]

    countingIndex = 0
    pbar = tqdm(total=100)
    while(countingIndex < Index_Effect):
        # Capture frame-by-frame
        ret, frame = videoIn.read()

        # if frame is read correctly ret is True
        if not ret:
            print("Stream End...")
            break

        if countingIndex < flagNumber[0]:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            sobelx = cv2.Sobel(gray,cv2.CV_8U,1,0,ksize=3)
            output = cv2.cvtColor(sobelx, cv2.COLOR_GRAY2RGB)
            videoOut.write(output)
        else:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            sobely = cv2.Sobel(gray,cv2.CV_8U,0,1,ksize=5)
            output = cv2.cvtColor(sobely, cv2.COLOR_GRAY2RGB)
            videoOut.write(output)

        countingIndex += 1
        if ((countingIndex/Index_Effect)*100) >= 100:
            pbar.update(100-pbar.n)
        else:
            pbar.update((((countingIndex/Index_Effect)*100)-pbar.n))
    pbar.close()

def drawCircleContour(frame, circles):
    color = (0, 255, 0)
    if circles is not None:
        circles = np.uint16(np.around(circles))
        for i in circles[0,:]:
            x, y, r = i[0], i[1], i[2]
            # Draw the circumference of the circle.
            cv2.circle(frame, (x, y), r, color, 2)
    return frame

def houghCircleDetection(videoIn, videoOut, fps):
    # 5 second
    Index_Effect = 10 * fps
    flagNumber = [2*fps,4*fps,5.5*fps,7*fps,8.5*fps]

    countingIndex = 0
    pbar = tqdm(total=100)
    while(countingIndex < Index_Effect):
        # Capture frame-by-frame
        ret, frame = videoIn.read()

        # if frame is read correctly ret is True
        if not ret:
            print("Stream End...")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rows = gray.shape[0]
        circles = 0 # initialize circles variable
        if countingIndex < flagNumber[0]:
            circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 2, rows / 16,
                                   param1=50, param2=10,
                                   minRadius=1, maxRadius=180)
        elif countingIndex < flagNumber[1]:
            circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, rows / 16,
                                   param1=50, param2=10,
                                   minRadius=1, maxRadius=180)

        elif countingIndex < flagNumber[2]:
            circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, rows / 6,
                                   param1=50, param2=10,
                                   minRadius=1, maxRadius=180)
        elif countingIndex < flagNumber[3]:
            circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, rows / 6,
                                   param1=50, param2=40,
                                   minRadius=1, maxRadius=180)

        elif countingIndex < flagNumber[4]:
            circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, rows / 6,
                                   param1=120, param2=40,
                                   minRadius=1, maxRadius=180)
        else:
            circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, rows / 6,
                                   param1=120, param2=40,
                                   minRadius=30, maxRadius=140)
        frame = drawCircleContour(frame, circles)
        videoOut.write(frame)
        countingIndex += 1
        if ((countingIndex/Index_Effect)*100) >= 100:
            pbar.update(100-pbar.n)
        else:
            pbar.update((((countingIndex/Index_Effect)*100)-pbar.n))
    pbar.close()


def templateMatching(videoIn, videoOut, fps):
    # 5 second
    Index_Effect = 5 * fps
    flagNumber = [2*fps]

    countingIndex = 0
    pbar = tqdm(total=100)
    while(countingIndex < Index_Effect):
        # Capture frame-by-frame
        ret, frame = videoIn.read()

        # if frame is read correctly ret is True
        if not ret:
            print("Stream End...")
            break

        lower = np.array([0, 0, 100], dtype = "uint8")
        upper = np.array([50, 56, 255], dtype = "uint8")
        # find the colors within the specified boundaries and apply
        # the mask
        mask = cv2.inRange(frame, lower, upper)
        output = cv2.bitwise_and(frame, frame, mask = mask)

        kernel = cv2.getStructuringElement(cv2.MORPH_CROSS,(5,5))
        opening = cv2.morphologyEx(output, cv2.MORPH_OPEN, kernel)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(10,10))
        closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)
        gray = cv2.cvtColor(closing,cv2.COLOR_BGR2GRAY)
        _,thresh = cv2.threshold(gray,20,255,cv2.THRESH_BINARY)
        contours,hierarchy = cv2.findContours(thresh,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) != 0:
            cnt = max(contours, key = cv2.contourArea)

            x,y,w,h = cv2.boundingRect(cnt)
            crop = closing[y:y+h,x:x+w]

            if countingIndex < flagNumber[0]:
                top_left = (x,y)
                bottom_right = (x + w, y + h)

                cv2.rectangle(frame,top_left, bottom_right, 255, 2)
                videoOut.write(frame)
            else:
                gray_crop = cv2.cvtColor(crop,cv2.COLOR_BGR2GRAY)

                # Apply template Matching
                res = cv2.matchTemplate(gray,gray_crop,cv2.TM_CCOEFF_NORMED)
                res = cv2.cvtColor(res,cv2.COLOR_GRAY2BGR)
                temp = cv2.resize(res,(1280,720))
                temp = img_as_ubyte(temp)
                videoOut.write(temp)
        countingIndex += 1
        if ((countingIndex/Index_Effect)*100) >= 100:
            pbar.update(100-pbar.n)
        else:
            pbar.update((((countingIndex/Index_Effect)*100)-pbar.n))
    pbar.close()

def objectColorChange(videoIn, videoOut, fps):
    # 6 second
    Index_Effect = 5.5 * fps
    colorChangeInterval = 0.3 * fps
    colorList = [(0,0,255),(0,127,255),(0,255,255),(0,255,0),(255,0,0),(130,0,75),(211,0,148)]
    flagNumber = [2*fps]

    countingIndex = 0
    pbar = tqdm(total=100)
    while(countingIndex < Index_Effect):
        # Capture frame-by-frame
        ret, frame = videoIn.read()

        # if frame is read correctly ret is True
        if not ret:
            print("Stream End...")
            break

        #   ******************************************
        #            Change Red object Color
        #   ******************************************
        red_low=np.array([0,0,100])
        red_high=np.array([50,56,255])

        mask=cv2.inRange(frame,red_low,red_high)

        frame[mask>0] = colorList[int((countingIndex % (1.5 * fps)) / colorChangeInterval)]

        #   ******************************************
        #          Turn image to Cartoon Style
        #   ******************************************

        gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        gray = cv2.medianBlur(gray, 5)
        edges = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 9)
        color = cv2.bilateralFilter(frame, 9, 200, 200)
        cartoon = cv2.bitwise_and(color, color, mask=edges)

        videoOut.write(cartoon)
        countingIndex += 1
        if ((countingIndex/Index_Effect)*100) >= 100:
            pbar.update(100-pbar.n)
        else:
            pbar.update((((countingIndex/Index_Effect)*100)-pbar.n))
    pbar.close()

def invisibleCloak(videoIn, videoOut, fps):

    # Creating a VideoCapture object
    cap = readVideo("./input/videoBackground.mp4")
    background = 0
    for i in range(60):
        # Capture frame-by-frame
        ret, background = cap.read()

        # if frame is read correctly ret is True
        if not ret:
            print("Stream End...")
            break

    background = cv2.resize(background,(1280,720))
    background = np.flip(background, axis=1)

    countingIndex = 0
    while(True):
        # Capture frame-by-frame
        ret, frame = videoIn.read()

        # if frame is read correctly ret is True
        if not ret:
            print("Stream End...")
            break

        # Laterally invert the image / flip the image
        frame  = np.flip(frame, axis=1)

        lower = np.array([0, 0, 0], dtype = "uint8")
        upper = np.array([75, 75, 75], dtype = "uint8")
        # find the colors within the specified boundaries and apply
        # the mask
        mask = cv2.inRange(frame, lower, upper)

        mask1 = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3,3),np.uint8))
        mask1 = cv2.morphologyEx(mask, cv2.MORPH_DILATE, np.ones((3,3),np.uint8))
        mask_invert = cv2.bitwise_not(mask1)

        res1 = cv2.bitwise_and(frame,frame,mask=mask_invert)

        # creating image showing static background frame pixels only for the masked region
        res2 = cv2.bitwise_and(background, background, mask = mask1)


        #Generating the final output
        final_output = cv2.addWeighted(res1,1,res2,1,0)
        videoOut.write(final_output)
        countingIndex += 1



def main(_):
    inputVideoPath = "./input/inputVideo.mp4"
    videoIn = readVideo(inputVideoPath)
    fps = videoIn.get(cv2.CAP_PROP_FPS)

    videoOut = getVideoWriter(videoIn,fps)

    VideoEffectFunctions = ["switchGrayBGR","blurBilateralGaussian","detectObjectRGBHSV","sobelEdgeDetection","houghCircleDetection","templateMatching","objectColorChange","invisibleCloak"]
    for i in VideoEffectFunctions:
        print("Processing " + i + "......")
        globals()[i](videoIn, videoOut, fps)

    # When everything done, release the capture
    videoIn.release()
    videoOut.release()

    os.system('ffmpeg -y -i ./input/music.mp3 -r 30 -i ./temporaryGeneratedFiles/outpy.avi  -filter:a aresample=async=1 -c:a flac -c:v copy ./input/temporaryGeneratedFiles/temp_output_mkv.mkv') # add mp3 to video
    os.system('rm -r -f ./output/outputVideo.mp4') # remove previous output video
    os.system('rm -r -f ./input/temporaryGeneratedFiles/temp_output_mp4.mp4') # remove previous temp output video
    os.system('ffmpeg -i ./input/temporaryGeneratedFiles/temp_output_mkv.mkv -strict -2 -c copy ./input/temporaryGeneratedFiles/temp_output_mp4.mp4') # turn mkv to mp4
    os.system('ffmpeg -y -i ./input/temporaryGeneratedFiles/temp_output_mp4.mp4 \
      -c:v libx264 -b:v 1500k -pass 1 \
      -b:a 128k -f mp4 /dev/null && \
      ffmpeg -i ./input/temporaryGeneratedFiles/temp_output_mp4.mp4 \
      -c:v libx264 -b:v 1500k -pass 2 \
      -b:a 128k ./output/outputVideo.mp4') # compress output video
    print('Processing Done.....')


if __name__ == "__main__":
    app.run(main)
