import cv2
import imutils
import numpy as np
import os

def skull_stripping(image, plot=False):
    
    # Convert the image to grayscale, and blur it slightly
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    # Threshold the image, then perform a series of erosions +
    # dilations to remove any small regions of noise
    thresh = cv2.threshold(gray, 45, 255, cv2.THRESH_BINARY)[1]
    thresh = cv2.erode(thresh, None, iterations=2)
    thresh = cv2.dilate(thresh, None, iterations=2)

    # Find contours in thresholded image, then grab the largest one
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    c = max(cnts, key=cv2.contourArea)
    

    # Find the extreme points
    extLeft = tuple(c[c[:, :, 0].argmin()][0])
    extRight = tuple(c[c[:, :, 0].argmax()][0])
    extTop = tuple(c[c[:, :, 1].argmin()][0])
    extBot = tuple(c[c[:, :, 1].argmax()][0])
    
    # crop new image out of the original image using the four extreme points (left, right, top, bottom)
    new_image = image[extTop[1]:extBot[1], extLeft[0]:extRight[0]]            
    
    return new_image


def reduce_noise(image):
    """
    Reduces noise in a grayscale image using Gaussian blur.
    Arguments:
        image: The input image (grayscale or color) as a NumPy array
    Returns: The image with reduced noise.
    """
    # Apply Gaussian Blur to reduce noise
    # ksize=(5, 5) is the kernel size and the value of 0 calculates the standard deviation automatically
    denoised_image = cv2.GaussianBlur(image, (5, 5), 0)
    return denoised_image


def pre_process_data(dir_list, noise_reduction, skull_strip, update_callback):
    
    image_width, image_height = 240, 240
    print("Noise Reduction:", noise_reduction)
    print("Skull Stripping:", skull_strip)
    
    dataset_dir = dir_list[0]
    pre_processed_data_dir = os.path.join(os.path.dirname(dataset_dir), 'pre_processed_data')
    
    # Delete previous processed data directory
    if os.path.exists(pre_processed_data_dir):
        os.system(f'rmdir "{pre_processed_data_dir}" /s /q')
        print("Previous pre-processed data directory found and deleted!")
    
    # create pre_processed_data directory if not exists
    if not os.path.exists(pre_processed_data_dir):
        os.makedirs(pre_processed_data_dir)
    
    for directory in dir_list:
        images_processed = 0
        print("Directory: ", directory)
        image_files = [f for f in os.listdir(directory) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        save_directory = os.path.join(pre_processed_data_dir, os.path.basename(os.path.normpath(directory)))
        print("Save directory: ", save_directory)
        
        if not os.path.exists(save_directory):
            os.makedirs(save_directory)
            
        update_callback(0, len(image_files))

        # Process each image in the directory
        # crop the brain and ignore the unnecessary rest part of the image
        # apply Gaussian blur to reduce noise if required
        # save the processed image to the new directory
        # update the progress bar and print the progress if provided

        print("Directory being Processed: ", directory)
        for filename in image_files:            
            image = cv2.imread(os.path.join(directory, filename))
            
            # resize image (Scaling)
            # image = cv2.resize(image, dsize=(image_width, image_height), interpolation=cv2.INTER_CUBIC)
            
            # normalize values and convert back to 8-bit for OpenCV functions
            image = (image / 255.0 * 255).astype(np.uint8)
            
            # crop the brain and ignore the unnecessary rest part of the image
            if skull_strip:
                image = skull_stripping(image, plot=False)
            if noise_reduction:
                image = reduce_noise(image)
            
            # Save the processed image to the new directory
            save_path = os.path.join(save_directory, filename)
            cv2.imwrite(save_path, image)  # Save as 8-bit image
            
            images_processed += 1
            print(f"Images Processed: {images_processed}/{len(image_files)} | Filename: {filename}", end='\r')
            update_callback(images_processed, len(image_files))

      