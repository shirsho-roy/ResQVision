import cv2
import os
import datetime
import math
import numpy as np
import time

output_dir = 'C:\PDLAB\outputs'
os.makedirs(output_dir, exist_ok=True)


def calculate_psnr(img1, img2):
    img1 = img1.astype('float64')/255
    img2 = img2.astype('float64')/255
    img1 = cv2.resize(img1, (400, 600))
    img2 = cv2.resize(img2, (400, 600))
    mse = np.mean((img1 - img2)**2)
    if mse == 0:
        return float('inf')
    return 20 * math.log10(255.0 / math.sqrt(mse))

def ssim(img1, img2):
    C1 = (0.01 * 255)**2
    C2 = (0.03 * 255)**2
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    img1 = cv2.resize(img1, (400, 600))
    img2 = cv2.resize(img2, (400, 600))
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())
    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1**2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2**2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                                            (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()

def calculate_ssim(img1, img2):
    img1 = cv2.resize(img1, (400, 600))
    img2 = cv2.resize(img2, (400, 600))
    if not img1.shape == img2.shape:
        raise ValueError('Input images must have the same dimensions.')
    if img1.ndim == 2:
        return ssim(img1, img2)
    elif img1.ndim == 3:
        if img1.shape[2] == 3:
            ssims = []
            for i in range(3):
                ssims.append(ssim(img1, img2))
            return np.array(ssims).mean()
        elif img1.shape[2] == 1:
            return ssim(np.squeeze(img1), np.squeeze(img2))
    else:
        raise ValueError('Wrong input image dimensions.')

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
    guided_filter = mean_a*im + mean_b
    return guided_filter

def Refine_Transmission(image, et):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray_image = np.float64(gray_image)/255
    r = 60
    eps = 0.0001
    transmission = Guidedfilter(gray_image, et, r, eps)
    return transmission

def Recover(im, t, A, tx=0.1):
    res = np.empty(im.shape, im.dtype)
    t = cv2.max(t, tx)
    for ind in range(0, 3):
        res[:, :, ind] = (im[:, :, ind]-A[0, ind])/t + A[0, ind]
    return res

def empty_last_few_images(folder, num_to_keep):
    files = os.listdir(folder)
    files.sort(key=lambda x: os.path.getctime(os.path.join(folder, x)))
    num_to_delete = max(0, len(files) - num_to_keep)
    for i in range(num_to_delete):
        file_to_delete = os.path.join(folder, files[i])
        os.remove(file_to_delete)

i = 0
empty_interval = 1
last_empty_time = time.time()

image_folder = 'inputs'
image_files = [f for f in os.listdir(image_folder) if f.endswith(('.jpg', '.jpeg', '.png'))]

for image_file in image_files:
    frame = cv2.imread(os.path.join(image_folder, image_file))

    I = frame.astype('float64') / 255
    dark = DarkChannel(I)
    A = AtmLight(I, dark)
    te = Estimate_transmission(I, A)
    t = Refine_Transmission(frame, te)
    dehazed_frame = Recover(I, t, A, 0.1)

    cv2.imshow('Original', frame)
    cv2.imshow('Dehazed', (dehazed_frame * 255).astype(np.uint8))

    if cv2.waitKey(1000) & 0xFF == ord('q'):
        break

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    num = '{:04d}'.format(i)
    original_frame_filename = f'{output_dir}/original_{num}.png'
    dehazed_frame_filename = f'{output_dir}/dehazed_{num}.png'

    cv2.imwrite(original_frame_filename, frame)
    cv2.imwrite(dehazed_frame_filename, (dehazed_frame * 255).astype(np.uint8))

    i += 1

    current_time = time.time()
    if current_time - last_empty_time >= empty_interval:
        last_empty_time = current_time
        empty_last_few_images(output_dir, 30)
   
