Real-Time Dehazing Model Documentation

Introduction:
This is a real-time dehazing model implemented using Python and OpenCV. Dehazing is a process to remove the haze or fog from images, improving visibility and enhancing image quality. This model captures frames from a webcam or processes images from a specified folder using dehazing algorithms and displays the original and dehazed images.

Dependencies:
- Python 3.x
- OpenCV (cv2)
- NumPy

Features:
1. Real-time dehazing: Processes images in real-time from a webcam or specified folder and applies dehazing algorithms.
2. Dark Channel Prior: Utilizes the dark channel prior method to estimate the atmospheric light and transmission.
3. Guided Filter: Applies guided filtering to refine the estimated transmission map.
4. Atmospheric Light Estimation: Estimates atmospheric light using the dark channel prior.
5. Transmission Map Estimation: Estimates the transmission map based on the dark channel prior and atmospheric light.
6. Recovery: Recovers the dehazed image using the estimated transmission map and atmospheric light.
7. Supports batch processing: Can process multiple images from a specified folder.

Usage:
1. Ensure that the necessary dependencies are installed (Python, OpenCV, NumPy).
2. Place the images to be processed in the 'inputs' folder.
3. Run the script.
4. Press 'q' to exit the program.

Documentation:
- The script processes images from the 'inputs' folder by default. To change the input folder, modify the 'image_folder' variable.
- The original and dehazed images are displayed in separate windows.
- Optionally, processed images can be saved to the 'outputs' folder. Adjust the 'output_dir' variable to change the output directory.
- Performance metrics such as PSNR and SSIM are calculated between the original and dehazed images.
- The number of images to keep in the output directory can be adjusted by modifying the 'empty_last_few_images' function.
- This model assumes a constant atmospheric light for simplicity.
- The performance of the dehazing algorithm may vary based on environmental conditions and image quality.
- Additional optimizations and improvements can be made for specific use cases.

