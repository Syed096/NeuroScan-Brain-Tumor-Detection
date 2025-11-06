from tensorflow.keras.preprocessing.image import ImageDataGenerator
import time
import os
import cv2 
import numpy as np

# Function to add noise to an image
def add_noise(image):
    noise_factor = 0.1 
    noise = np.random.randn(*image.shape) * noise_factor
    noisy_image = image + noise
    noisy_image = np.clip(noisy_image, 0., 1.)  # Ensure pixel values are within [0, 1]
    return noisy_image

# Function to perform data augmentation
def augment_data(file_dir, n_generated_samples, save_to_dir, rotation, scaling, noise_injection, update_callback):
    data_gen_args = {}

    # Apply rotation if selected
    if rotation:
        data_gen_args['rotation_range'] = 10

    # Normalize pixel values if scaling is selected
    if scaling:
        data_gen_args['rescale'] = 1/255
    
    # Initialize ImageDataGenerator with preprocessing function for noise injection
    data_gen = ImageDataGenerator(
        **data_gen_args,
        preprocessing_function=add_noise if noise_injection else None
    )

    images_processed = 0
    # List all image files (grayscale)
    image_files = [f for f in os.listdir(file_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    print("Directory being processed: ", file_dir)
    
    for filename in image_files:
        # Read image as grayscale
        image = cv2.imread(os.path.join(file_dir, filename), cv2.IMREAD_GRAYSCALE)
        
        if scaling:
            image = image / 255.0  # Ensure the image is normalized
            image_width, image_height = image.shape
            image = cv2.resize(image, dsize=(image_width, image_height), interpolation=cv2.INTER_CUBIC)
        
        # Reshape for generator
        image = image.reshape((1,) + image.shape + (1,))  # (batch_size, height, width, channels)

        # Save prefix for the augmented files
        save_prefix = 'aug_' + filename[:-4]
        
        i = 0
        for batch in data_gen.flow(x=image, batch_size=1, save_to_dir=save_to_dir, 
                                   save_prefix=save_prefix, save_format='jpg'):
            i += 1
            if i >= n_generated_samples:
                break
        
        images_processed += 1
        print(f'Images Processed: {images_processed}/{len(image_files)}', end='\r')
        update_callback(images_processed, len(image_files))

# Main function for performing data augmentation
def perform_data_augmentation(file_dir, update_callback, rotation, scaling, noise_injection):
    start_time = time.time()
    print("Path Received in Data-Augmentation: ", file_dir)
    
    # Paths for reading images
    yes_path = os.path.join(file_dir, 'yes')
    no_path = os.path.join(file_dir, 'no')
    
    if 'pre_processed_data' in file_dir:
        file_dir = os.path.dirname(file_dir)
    
    # Paths for saving augmented data
    augmented_data_path = os.path.join(file_dir, 'augmented_data')
    print(f"Augmented data path: {augmented_data_path}")
    
    # Delete previous augmented data directory
    if os.path.exists(augmented_data_path):
        os.system(f'rmdir "{augmented_data_path}" /s /q')    
        print("Previous augmented data directory found and deleted!")
        
    # Create directories if they don't exist
    if not os.path.exists(augmented_data_path):
        os.mkdir(augmented_data_path)
    if not os.path.exists(os.path.join(augmented_data_path, 'yes')):
        os.mkdir(os.path.join(augmented_data_path, 'yes'))
    if not os.path.exists(os.path.join(augmented_data_path, 'no')):
        os.mkdir(os.path.join(augmented_data_path, 'no'))
    
    print("Rotation:", rotation)
    print("Scaling:", scaling)
    print("Noise Injection:", noise_injection)
    
    # Perform augmentation for images labeled 'yes' (tumorous)
    augment_data(file_dir=yes_path, n_generated_samples=6, save_to_dir=os.path.join(augmented_data_path, 'yes'),
                 rotation=rotation, scaling=scaling, noise_injection=noise_injection, update_callback=update_callback)
    
    # Perform augmentation for images labeled 'no' (non-tumorous)
    augment_data(file_dir=no_path, n_generated_samples=9, save_to_dir=os.path.join(augmented_data_path, 'no'),
                 rotation=rotation, scaling=scaling, noise_injection=noise_injection, update_callback=update_callback)

    end_time = time.time()
    execution_time = end_time - start_time
    print(f'Augmentation process completed in {execution_time:.2f} seconds.')

# Example usage
# if __name__ == '__main__':
#     path = os.path.join(os.path.dirname(__file__), 'Brain_Tumor', 'Dataset')
#     perform_data_augmentation(file_dir=path, rotation=True, scaling=False, noise_injection=False)
