import cv2
import os
import datetime
import math
import numpy as np
import time 
from PIL import Image

# Define the function for dehazing
def dehaze():
    # Function definitions
    def DarkChannel(im):
        b, g, r = cv2.split(im)
        dc = cv2.min(cv2.min(r, g), b)
        kernel = np.ones((5, 5), np.uint8)
        dark = cv2.erode(dc, kernel)
        return dark

    def AtmLight(im, dark):
        [h, w] = im.shape[:2]
        imsz = h*w
        numpx = int(max(math.floor(imsz/1000), 1))
        darkvec = dark.reshape(imsz)
        imvec = im.reshape(imsz, 3)
        indices = darkvec.argsort()
        indices = indices[(imsz-numpx)::]
        atmsum = np.zeros([1, 3])
        for ind in range(1, numpx):
            atmsum = atmsum + imvec[indices[ind]]
        A = atmsum / numpx
        return A

    def Estimate_transmission(image, A):
        omega = 0.95
        image3 = np.empty(image.shape, image.dtype)
        for ind in range(0, 3):
            image3[:, :, ind] = image[:, :, ind]/A[0, ind]
        transmission = 1 - omega*DarkChannel(image3)
        return transmission

    def Guidedfilter(im, p, r, eps):  
        mean_I = cv2.boxFilter(im, cv2.CV_64F, (r, r))
        mean_p = cv2.boxFilter(p, cv2.CV_64F, (r, r))
        mean_Ip = cv2.boxFilter(im*p, cv2.CV_64F, (r, r))
        cov_Ip = mean_Ip - mean_I*mean_p
        mean_II = cv2.boxFilter(im*im, cv2.CV_64F, (r, r))
        var_I = mean_II - mean_I*mean_I
        a = cov_Ip/(var_I + eps)
        b = mean_p - a*mean_I
        mean_a = cv2.boxFilter(a, cv2.CV_64F, (r, r))
        mean_b = cv2.boxFilter(b, cv2.CV_64F, (r, r))
        filter = mean_a*im + mean_b
        return filter

    def Refine_Transmission(image, et):
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray_image = np.float64(gray_image)/255
        r = 60
        eps = 0.0001
        t = Guidedfilter(gray_image, et, r, eps)
        return t

    def Recover(im, t, A, tx=0.1):
        res = np.empty(im.shape, im.dtype)
        t = cv2.max(t, tx)
        for ind in range(0, 3):
            res[:, :, ind] = (im[:, :, ind]-A[0, ind])/t + A[0, ind]
        return res

    # Open a connection to the webcam (use the appropriate camera index)
    cap = cv2.VideoCapture(0)

    # Check if the camera opened successfully
    if not cap.isOpened():
        return "Error: Could not open camera."

    while True:
        ret, frame = cap.read()  # Capture a frame from the webcam

        if not ret:
            return "Error: Could not read frame."

        # Apply dehazing to the frame
        I = frame.astype('float64') / 255
        dark = DarkChannel(I)
        A = AtmLight(I, dark)
        te = Estimate_transmission(I, A)
        t = Refine_Transmission(frame, te)
        dehazed_frame = Recover(I, t, A, 0.1)

        # Convert the frames to PIL images
        original_frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        dehazed_frame = Image.fromarray(cv2.cvtColor((dehazed_frame * 255).astype(np.uint8), cv2.COLOR_BGR2RGB))

        yield original_frame, dehazed_frame

# Deploy on Gradio
import gradio as gr

iface = gr.Interface(dehaze, 
                     inputs=None, 
                     outputs=["image", "image"], 
                     title="Real-time Dehazing",
                     description="This app dehazes live video from your webcam.")
iface.launch()
