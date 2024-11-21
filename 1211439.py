'''
- filter's  MSE TIME GUSAIN 
    - Name: leyan burait 
    - ID: 1211439
    - Section: 2
'''

import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import time
import csv
import pandas as pd

########################################################################################################################################
#                                                                Load Images                                                           #
########################################################################################################################################

'''
--> In this part of the code, we load images through the `load_image(image_path)` function, 
which takes the image path and then checks the existence of the image `the validity of the path`.
If the image exists, it returns the image. Otherwise, it shows an error and returns Noon.

--> load three images:
A picture without details (simple picture): no_detail_image.jpeg
A picture with medium details: medium_detail_image.jpeg
A picture with details: detailed_image.jpeg

'''

# A function to load image:
def load_image(image_path):
    try:
        # Load the image in the gary scale:
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        # Check if the image read succussfuly:
        if image is None:
            raise ValueError(f"Failed to load image: {image_path}")
        # Return the image after resize it (to be sure that all image have the same size):
        return cv2.resize(image, (0, 0), fx=0.5, fy=0.5)
    except Exception as e:
        print(f"An error occurred while loading the image: {e}")
        return None

# Load the three images:
detailed_image = load_image('highdetailed_image.jpg')
medium_detail_image = load_image('wasat.jpg')
no_detail_image = load_image('leasdetail_img.jpg')

'''
--> Ensure that all images have been loaded successfully:
If the all images loaded successfully then 
    statrt work with them `add noise and applying different filters, ..etc`.
else 
    display a message to check the images path.
'''

if detailed_image is not None and medium_detail_image is not None and no_detail_image is not None:
    
    ####################################################################################################################################
    #                                                              Add Noise                                                           #
    ####################################################################################################################################
   
    '''
    --> In this part of the code, we add different noises to images at different levels (low, medium, high) through the functions:
    `add_gaussian_noise(image, mean=0, sigma_level='medium')` and `add_salt_and_pepper_noise(image, intensity='medium')`.

        - The function `add_gaussian_noise(image, mean=0, sigma_level='medium')`:
          it adds gaussian noise to the image at different levels that are specified so that the default level is considered the medium
          and returns the image with adding noise to it.

        - The function `add_salt_and_pepper_noise(image, intensity='medium')`:
          it adds salt and pepper noise to the image at different levels that are specified so that the default level is considered 
          the medium and returns the image with adding noise to it.
    '''
    
    def add_gaussian_noise(image, mean=0, sigma_level='medium'):

        # Through the level (low, medium, high) we determine the value of sigma:
        if sigma_level == 'low':
            sigma = 10
        elif sigma_level == 'medium':
            sigma = 50
        elif sigma_level == 'high':
            sigma = 150
        else:
            raise ValueError("Invalid sigma level. Choose 'low', 'medium', or 'high'.")
        
        # Add gaussian noise to the image:
        gaussian_noise = np.random.normal(mean, sigma, image.shape).astype(np.uint8)
        noisy_image = cv2.add(image, gaussian_noise)

        # Return the image after add the gaussian noise to it:
        return noisy_image 

    def add_salt_and_pepper_noise(image, intensity='medium'):

        # Through the level (low, medium, high) we determine the values of salt prob and pepper prob:
        if intensity == 'low':
            salt_prob, pepper_prob = 0.01, 0.01
        elif intensity == 'medium':
            salt_prob, pepper_prob = 0.1, 0.1
        elif intensity == 'high':
            salt_prob, pepper_prob = 0.5, 0.5
        else:
            raise ValueError("Invalid intensity level. Choose 'low', 'medium', or 'high'.")
        
        # Add salt and pepper noise to the image:
        noisy_image = np.copy(image)
        total_pixels = image.size
        num_salt = int(salt_prob * total_pixels)
        num_pepper = int(pepper_prob * total_pixels)

        # Add salt noise (white points):
        coords = [np.random.randint(0, i, num_salt) for i in image.shape]
        noisy_image[coords[0], coords[1]] = 255

        # Add pepper noise (blace points):
        coords = [np.random.randint(0, i, num_pepper) for i in image.shape]
        noisy_image[coords[0], coords[1]] = 0
        
        # Return the image after add the salt and pepper noise to it:
        return noisy_image

    # Add noise to images:

    # List of all the noisy images & it's info:
    images = []
    
    '''
    1- Add gussian noise for the different images `detailed_image, medium_detail_image, no_detail_image` 
    with different intensity levels `low, medium, high`:
    '''
    gaussian_noisy_low_detailed_image = add_gaussian_noise(detailed_image, mean=0, sigma_level='low')
    images.append((gaussian_noisy_low_detailed_image, 'Detailed Image', 'Gaussian', 'low', detailed_image))
    gaussian_noisy_medium_detailed_image = add_gaussian_noise(detailed_image, mean=0, sigma_level='medium')
    images.append((gaussian_noisy_medium_detailed_image, 'Detailed Image', 'Gaussian', 'medium', detailed_image))
    gaussian_noisy_high_detailed_image = add_gaussian_noise(detailed_image, mean=0, sigma_level='high')
    images.append((gaussian_noisy_high_detailed_image, 'Detailed Image', 'Gaussian', 'high', detailed_image))

    gaussian_noisy_low_medium_detail_image = add_gaussian_noise(medium_detail_image, mean=0, sigma_level='low')
    images.append((gaussian_noisy_low_medium_detail_image, 'Medium Detail Image', 'Gaussian', 'low', medium_detail_image))
    gaussian_noisy_medium_medium_detail_image = add_gaussian_noise(medium_detail_image, mean=0, sigma_level='medium')
    images.append((gaussian_noisy_medium_medium_detail_image, 'Medium Detail Image', 'Gaussian', 'medium', medium_detail_image))
    gaussian_noisy_high_medium_detail_image = add_gaussian_noise(medium_detail_image, mean=0, sigma_level='high')
    images.append((gaussian_noisy_high_medium_detail_image, 'Medium Detail Image', 'Gaussian', 'high', medium_detail_image))

    gaussian_noisy_low_no_detail_image = add_gaussian_noise(no_detail_image, mean=0, sigma_level='low')
    images.append((gaussian_noisy_low_no_detail_image, 'No Detail Image', 'Gaussian', 'low', no_detail_image))
    gaussian_noisy_medium_no_detail_image = add_gaussian_noise(no_detail_image, mean=0, sigma_level='medium')
    images.append((gaussian_noisy_medium_no_detail_image, 'No Detail Image', 'Gaussian', 'medium', no_detail_image))
    gaussian_noisy_high_no_detail_image = add_gaussian_noise(no_detail_image, mean=0, sigma_level='high')
    images.append((gaussian_noisy_high_no_detail_image, 'No Detail Image', 'Gaussian', 'high', no_detail_image))

    '''
    1- Add salt and pepper noise for the different images `detailed_image, medium_detail_image, no_detail_image` 
    with different intensity levels `low, medium, high`:
    '''
    salt_pepper_noisy_low_detailed_image = add_salt_and_pepper_noise(detailed_image, intensity='low')
    images.append((salt_pepper_noisy_low_detailed_image, 'Detailed Image', 'Salt and Pepper', 'low', detailed_image))
    salt_pepper_noisy_medium_detailed_image = add_salt_and_pepper_noise(detailed_image, intensity='medium')
    images.append((salt_pepper_noisy_medium_detailed_image, 'Detailed Image', 'Salt and Pepper', 'medium', detailed_image))
    salt_pepper_noisy_high_detailed_image = add_salt_and_pepper_noise(detailed_image, intensity='high')
    images.append((salt_pepper_noisy_high_detailed_image, 'Detailed Image', 'Salt and Pepper', 'high', detailed_image))

    salt_pepper_noisy_low_medium_detail_image = add_salt_and_pepper_noise(medium_detail_image, intensity='low')
    images.append((salt_pepper_noisy_low_medium_detail_image, 'Medium Detail Image', 'Salt and Pepper', 'low', medium_detail_image))
    salt_pepper_noisy_medium_medium_detail_image = add_salt_and_pepper_noise(medium_detail_image, intensity='medium')
    images.append((salt_pepper_noisy_medium_medium_detail_image, 'Medium Detail Image', 'Salt and Pepper', 'medium', medium_detail_image))
    salt_pepper_noisy_high_medium_detail_image = add_salt_and_pepper_noise(medium_detail_image, intensity='high')
    images.append((salt_pepper_noisy_high_medium_detail_image, 'Medium Detail Image', 'Salt and Pepper', 'high', medium_detail_image))

    salt_pepper_noisy_low_no_detail_image = add_salt_and_pepper_noise(no_detail_image, intensity='low')
    images.append((salt_pepper_noisy_low_no_detail_image, 'No Detail Image', 'Salt and Pepper', 'low', no_detail_image))
    salt_pepper_noisy_medium_no_detail_image = add_salt_and_pepper_noise(no_detail_image, intensity='medium')
    images.append((salt_pepper_noisy_medium_no_detail_image, 'No Detail Image', 'Salt and Pepper', 'medium', no_detail_image))
    salt_pepper_noisy_high_no_detail_image = add_salt_and_pepper_noise(no_detail_image, intensity='high')
    images.append((salt_pepper_noisy_high_no_detail_image, 'No Detail Image', 'Salt and Pepper', 'high', no_detail_image))
    
    ####################################################################################################################################
    #                                            Plot the noisy images & Save them in PDF file                                         #
    ####################################################################################################################################

    '''
    --> This part of the code displays images, both original images and images with Gaussian Noise and Salt and Paper Noise 
    and saves them in PDF file.
    '''
    def save_plot_images(original_image, gaussian_noisy_image, salt_pepper_noisy_image, title, level,  pdf):
        plt.figure(figsize=(15, 5))
        plt.suptitle(f'{title} image after apply the noise `Gaussian noise, Salt-and-Pepper noise` with {level} level')
        
        # Plot the original image:
        plt.subplot(1, 3, 1)
        plt.imshow(original_image, cmap='gray')
        plt.title(f'{title} - Original')
        plt.axis('off')

        # Plot the imgae with Gaussian noise:
        plt.subplot(1, 3, 2)
        plt.imshow(gaussian_noisy_image, cmap='gray')
        plt.title(f'{title} - Gaussian Noise')
        plt.axis('off')

        # Plot the image with Salt-and-Pepper noise:
        plt.subplot(1, 3, 3)
        plt.imshow(salt_pepper_noisy_image, cmap='gray')
        plt.title(f'{title} - Salt-and-Pepper Noise')
        plt.axis('off')

        # Save the shape to PDF if there is any ploting, else display a message there is no figurs:
        if plt.get_fignums():
            pdf.savefig()
            plt.close()
        else:
            print(f"Warning: No figure to save for {title}")

    # Create PDF and save figures:
    noise_pdf_file_name = 'images_with_noises.pdf'
    with PdfPages(noise_pdf_file_name) as pdf:
        save_plot_images(no_detail_image, gaussian_noisy_low_no_detail_image, salt_pepper_noisy_low_no_detail_image, 'No Details', 'low', pdf)
        save_plot_images(no_detail_image, gaussian_noisy_medium_no_detail_image, salt_pepper_noisy_medium_no_detail_image, 'No Details', 'medium', pdf)
        save_plot_images(no_detail_image, gaussian_noisy_high_no_detail_image, salt_pepper_noisy_high_no_detail_image, 'No Details', 'high', pdf)

        save_plot_images(medium_detail_image, gaussian_noisy_low_medium_detail_image, salt_pepper_noisy_low_medium_detail_image, 'Medium Details', 'low', pdf)
        save_plot_images(medium_detail_image, gaussian_noisy_medium_medium_detail_image, salt_pepper_noisy_medium_medium_detail_image, 'Medium Details', 'medium', pdf)
        save_plot_images(medium_detail_image, gaussian_noisy_high_medium_detail_image, salt_pepper_noisy_high_medium_detail_image, 'Medium Details', 'high', pdf)

        save_plot_images(detailed_image, gaussian_noisy_low_detailed_image, salt_pepper_noisy_low_detailed_image, 'Detailed', 'low', pdf)
        save_plot_images(detailed_image, gaussian_noisy_medium_detailed_image, salt_pepper_noisy_medium_detailed_image, 'Detailed', 'medium', pdf)
        save_plot_images(detailed_image, gaussian_noisy_high_detailed_image, salt_pepper_noisy_high_detailed_image, 'Detailed', 'high', pdf)

    # Display a message that the result saved to the PDF file:
    print("The figures are saved in a file: {}".format(noise_pdf_file_name))

    ####################################################################################################################################
    #                                                     Build The Filters Functions                                                  #
    ####################################################################################################################################

    '''
    --> In this part of the code I build a set of simple and advanced filters to be applied to the blurry images. 
        * Simple filters include: Box filter, Gaussian filter, Median filter. 
        * Advanced filters include: Adaptive mean filter, Adaptive median filter, Bilateral filter. 
        
    Filters are applied using functions:
        * `apply_box_filter(noisy_image, k)` --> function applies a box filter to the noisy image
        * `apply_gaussian_filter(noisy_image, k)` --> function applies a Gaussian filter to the noisy image
        * `apply_median_filter(noisy_image, k)` --> function applies a median filter to the noisy image
        * `apply_adaptive_mean_filter(noisy_image, k)` --> function applies an adaptive mean filter to the noisy image
        * `apply_adaptive_median_filter(noisy_image, k)` --> function applies an adaptive median filter to the noisy image
        * `apply_bilateral_filter(noisy_image, k)` --> function applies a bilateral filter to the noisy image
    '''
    # Simple filters:
    def apply_box_filter(noisy_image, k, Time):
        # Find the start time:
        start_time = time.time()
        # Applied the filter:
        box_filtered_image = cv2.blur(noisy_image, (k, k))
        # Find the end time:
        end_time = time.time()
        # Calculate the elapsed time:
        elapsed_time = end_time - start_time
        # Store the elapsed time:
        Time[k] = elapsed_time
        # Return the image after applying the box filter on it:
        return box_filtered_image

    def apply_gaussian_filter(noisy_image, k, Time):
        # Find the start time:
        start_time = time.time()
        # Applied the filter:
        gaussian_filtered_image = cv2.GaussianBlur(noisy_image, (k, k), 0)
        # Find the end time:
        end_time = time.time()
        # Calculate the elapsed time:
        elapsed_time = end_time - start_time
        # Store the elapsed time:
        Time[k] = elapsed_time
        # Return the image after applying the gaussian filter on it:
        return gaussian_filtered_image

    def apply_median_filter(noisy_image, k, Time):
        # Find the start time:
        start_time = time.time()
        # Applied the filter:
        median_filtered_image = cv2.medianBlur(noisy_image, k)
        # Find the end time:
        end_time = time.time()
        # Calculate the elapsed time:
        elapsed_time = end_time - start_time
        # Store the elapsed time:
        Time[k] = elapsed_time
        # Return the image after applying the median filter on it:
        return median_filtered_image

    # Advanced filters:
    def apply_adaptive_mean_filter(noisy_image, k, Time):
        # Find the start time:
        start_time = time.time()
        # Applied the filter:
        adaptive_mean_filtered_image = cv2.adaptiveThreshold(noisy_image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, k if k % 2 != 0 else k + 1, 5)
        # Find the end time:
        end_time = time.time()
        # Calculate the elapsed time:
        elapsed_time = end_time - start_time
        # Store the elapsed time:
        Time[k] = elapsed_time
        # Return the image after applying the mean filter on it:
        return adaptive_mean_filtered_image

    def apply_adaptive_median_filter(noisy_image, k, Time):
        # Find the start time:
        start_time = time.time()

        # Applied the filter:

        # Create a copy of the image to save the result:
        adaptive_median_filtered_image = noisy_image.copy()

        # Get image dimensions:
        rows, cols = noisy_image.shape

        # Scroll through each pixel in the image:
        for i in range(rows):
            for j in range(cols):
                # Defining kernel boundaries:
                r_start = max(0, i - k // 2)
                r_end = min(rows, i + k // 2 + 1)
                c_start = max(0, j - k // 2)
                c_end = min(cols, j + k // 2 + 1)

                # Extract kernel window:
                kernel_window = noisy_image[r_start:r_end, c_start:c_end]

                # Broker Account, Minimum, Maximum:
                median = np.median(kernel_window)
                min_val = np.min(kernel_window)
                max_val = np.max(kernel_window)

                # Check adaptive filter conditions:
                if min_val < median < max_val:
                    if min_val < noisy_image[i, j] < max_val:
                        adaptive_median_filtered_image[i, j] = noisy_image[i, j]
                    else:
                        adaptive_median_filtered_image[i, j] = median
                else:
                    adaptive_median_filtered_image[i, j] = median

        # Find the end time:
        end_time = time.time()
        # Calculate the elapsed time:
        elapsed_time = end_time - start_time
        # Store the elapsed time:
        Time[k] = elapsed_time
        # Return the image after applying the  filter on it:
        return adaptive_median_filtered_image
    
    def apply_bilateral_filter(noisy_image, k, Time):
        # Find the start time:
        start_time = time.time()
        # Applied the filter:
        bilateral_filtered_image = cv2.bilateralFilter(noisy_image, d=k, sigmaColor=75, sigmaSpace=75)
        # Find the end time:
        end_time = time.time()
        # Calculate the elapsed time:
        elapsed_time = end_time - start_time
        # Store the elapsed time:
        Time[k] = elapsed_time
        # Return the image after applying the bilateral filter on it:
        return bilateral_filtered_image

    ####################################################################################################################################
    #                                                Build Measure performance  Functions                                              #
    ####################################################################################################################################

    '''
    --> In this part of the code I build a functions to measure performance, MSE, PSNR, and Edge preservation:
    '''

    def calculate_mse(original, filtered):
        return np.mean((original - filtered) ** 2)

    def calculate_psnr(mse):
        if mse == 0:
            return float('inf')
        return 20 * np.log10(255.0 / np.sqrt(mse))

    # Apply Canny edge detector:
    def apply_edge_detection(image):
        return cv2.Canny(image, 100, 200)

    def calculate_fom(edges_original, edges_filtered):
        """
        Pratt's Figure of Merit (FOM) for edge comparison.
        Returns a value between 0 (no match) and 1 (perfect match).
        """
        # تأكد من أن الصور الثنائية
        edges_original = edges_original.astype(bool)
        edges_filtered = edges_filtered.astype(bool)
    
        # احسب التداخل بين الحواف
        intersection = np.logical_and(edges_original, edges_filtered)
        union = np.logical_or(edges_original, edges_filtered)

        # حساب نسبة التداخل
        fom_score = np.sum(intersection) / np.sum(union)
        return fom_score

    # Different kernel sizes:
    kernel_sizes = [3, 9, 15]

    # Dictionary to save the elapsed time:
    calculate_time = {
        'Detailed Image':{
            'Gaussian':{
                'low':{ 'Box':{ 3:0, 9:0, 15:0 }, 'Gaussian':{ 3:0, 9:0, 15:0 }, 'Median':{  3:0,  9:0,  15:0 }, 
                       'Adaptive Mean':{ 3:0, 9:0, 15:0 }, 'Adaptive Median':{  3:0, 9:0, 15:0 }, 'Bilateral':{ 3:0, 9:0, 15:0 }, },
                
                'medium':{ 'Box':{ 3:0, 9:0, 15:0 }, 'Gaussian':{ 3:0, 9:0, 15:0 }, 'Median':{  3:0,  9:0,  15:0 }, 
                       'Adaptive Mean':{ 3:0, 9:0, 15:0 }, 'Adaptive Median':{  3:0, 9:0, 15:0 }, 'Bilateral':{ 3:0, 9:0, 15:0 }, },
                
                'high':{ 'Box':{ 3:0, 9:0, 15:0 }, 'Gaussian':{ 3:0, 9:0, 15:0 }, 'Median':{  3:0,  9:0,  15:0 }, 
                       'Adaptive Mean':{ 3:0, 9:0, 15:0 }, 'Adaptive Median':{  3:0, 9:0, 15:0 }, 'Bilateral':{ 3:0, 9:0, 15:0 }, }
            },
            'Salt and Pepper':{
                'low':{ 'Box':{ 3:0, 9:0, 15:0 }, 'Gaussian':{ 3:0, 9:0, 15:0 }, 'Median':{  3:0,  9:0,  15:0 }, 
                       'Adaptive Mean':{ 3:0, 9:0, 15:0 }, 'Adaptive Median':{  3:0, 9:0, 15:0 }, 'Bilateral':{ 3:0, 9:0, 15:0 }, },
                
                'medium':{ 'Box':{ 3:0, 9:0, 15:0 }, 'Gaussian':{ 3:0, 9:0, 15:0 }, 'Median':{  3:0,  9:0,  15:0 }, 
                       'Adaptive Mean':{ 3:0, 9:0, 15:0 }, 'Adaptive Median':{  3:0, 9:0, 15:0 }, 'Bilateral':{ 3:0, 9:0, 15:0 }, },
                
                'high':{ 'Box':{ 3:0, 9:0, 15:0 }, 'Gaussian':{ 3:0, 9:0, 15:0 }, 'Median':{  3:0,  9:0,  15:0 }, 
                       'Adaptive Mean':{ 3:0, 9:0, 15:0 }, 'Adaptive Median':{  3:0, 9:0, 15:0 }, 'Bilateral':{ 3:0, 9:0, 15:0 }, }
            }
        },
        'Medium Detail Image':{
            'Gaussian':{
                'low':{ 'Box':{ 3:0, 9:0, 15:0 }, 'Gaussian':{ 3:0, 9:0, 15:0 }, 'Median':{  3:0,  9:0,  15:0 }, 
                       'Adaptive Mean':{ 3:0, 9:0, 15:0 }, 'Adaptive Median':{  3:0, 9:0, 15:0 }, 'Bilateral':{ 3:0, 9:0, 15:0 }, },
                
                'medium':{ 'Box':{ 3:0, 9:0, 15:0 }, 'Gaussian':{ 3:0, 9:0, 15:0 }, 'Median':{  3:0,  9:0,  15:0 }, 
                       'Adaptive Mean':{ 3:0, 9:0, 15:0 }, 'Adaptive Median':{  3:0, 9:0, 15:0 }, 'Bilateral':{ 3:0, 9:0, 15:0 }, },
                
                'high':{ 'Box':{ 3:0, 9:0, 15:0 }, 'Gaussian':{ 3:0, 9:0, 15:0 }, 'Median':{  3:0,  9:0,  15:0 }, 
                       'Adaptive Mean':{ 3:0, 9:0, 15:0 }, 'Adaptive Median':{  3:0, 9:0, 15:0 }, 'Bilateral':{ 3:0, 9:0, 15:0 }, }
            },
            'Salt and Pepper':{
                'low':{ 'Box':{ 3:0, 9:0, 15:0 }, 'Gaussian':{ 3:0, 9:0, 15:0 }, 'Median':{  3:0,  9:0,  15:0 }, 
                       'Adaptive Mean':{ 3:0, 9:0, 15:0 }, 'Adaptive Median':{  3:0, 9:0, 15:0 }, 'Bilateral':{ 3:0, 9:0, 15:0 }, },
                
                'medium':{ 'Box':{ 3:0, 9:0, 15:0 }, 'Gaussian':{ 3:0, 9:0, 15:0 }, 'Median':{  3:0,  9:0,  15:0 }, 
                       'Adaptive Mean':{ 3:0, 9:0, 15:0 }, 'Adaptive Median':{  3:0, 9:0, 15:0 }, 'Bilateral':{ 3:0, 9:0, 15:0 }, },
                
                'high':{ 'Box':{ 3:0, 9:0, 15:0 }, 'Gaussian':{ 3:0, 9:0, 15:0 }, 'Median':{  3:0,  9:0,  15:0 }, 
                       'Adaptive Mean':{ 3:0, 9:0, 15:0 }, 'Adaptive Median':{  3:0, 9:0, 15:0 }, 'Bilateral':{ 3:0, 9:0, 15:0 }, }
            }
        },
        'No Detail Image':{
            'Gaussian':{
                'low':{ 'Box':{ 3:0, 9:0, 15:0 }, 'Gaussian':{ 3:0, 9:0, 15:0 }, 'Median':{  3:0,  9:0,  15:0 }, 
                       'Adaptive Mean':{ 3:0, 9:0, 15:0 }, 'Adaptive Median':{  3:0, 9:0, 15:0 }, 'Bilateral':{ 3:0, 9:0, 15:0 }, },
                
                'medium':{ 'Box':{ 3:0, 9:0, 15:0 }, 'Gaussian':{ 3:0, 9:0, 15:0 }, 'Median':{  3:0,  9:0,  15:0 }, 
                       'Adaptive Mean':{ 3:0, 9:0, 15:0 }, 'Adaptive Median':{  3:0, 9:0, 15:0 }, 'Bilateral':{ 3:0, 9:0, 15:0 }, },
                
                'high':{ 'Box':{ 3:0, 9:0, 15:0 }, 'Gaussian':{ 3:0, 9:0, 15:0 }, 'Median':{  3:0,  9:0,  15:0 }, 
                       'Adaptive Mean':{ 3:0, 9:0, 15:0 }, 'Adaptive Median':{  3:0, 9:0, 15:0 }, 'Bilateral':{ 3:0, 9:0, 15:0 }, }
            },
            'Salt and Pepper':{
                'low':{ 'Box':{ 3:0, 9:0, 15:0 }, 'Gaussian':{ 3:0, 9:0, 15:0 }, 'Median':{  3:0,  9:0,  15:0 }, 
                       'Adaptive Mean':{ 3:0, 9:0, 15:0 }, 'Adaptive Median':{  3:0, 9:0, 15:0 }, 'Bilateral':{ 3:0, 9:0, 15:0 }, },
                
                'medium':{ 'Box':{ 3:0, 9:0, 15:0 }, 'Gaussian':{ 3:0, 9:0, 15:0 }, 'Median':{  3:0,  9:0,  15:0 }, 
                       'Adaptive Mean':{ 3:0, 9:0, 15:0 }, 'Adaptive Median':{  3:0, 9:0, 15:0 }, 'Bilateral':{ 3:0, 9:0, 15:0 }, },
                
                'high':{ 'Box':{ 3:0, 9:0, 15:0 }, 'Gaussian':{ 3:0, 9:0, 15:0 }, 'Median':{  3:0,  9:0,  15:0 }, 
                       'Adaptive Mean':{ 3:0, 9:0, 15:0 }, 'Adaptive Median':{  3:0, 9:0, 15:0 }, 'Bilateral':{ 3:0, 9:0, 15:0 }, }
            }
        }
    }

    # List to store the result data:
    results_data = []

    # Fuction to store the result in the results_data list:
    def log_results(filter_name, kernel_size, noise_type, noise_level, mse, psnr, elapsed_time, fom_score):
        results_data.append({
            "Filter": filter_name,
            "Kernel Size": kernel_size,
            "Noise Type": noise_type,
            "Noise Level": noise_level,
            "MSE": mse,
            "PSNR": psnr,
            "Time (s)": elapsed_time,
            "FOM":fom_score
        })

    def evaluate_filters(original, noisy_image, title, noise_type, noise_level, filter_name, filtered_images, kernel_sizes, Time, pdf):
        for filtered_image, k in zip(filtered_images, kernel_sizes):

            # Calculate MSE & PSNR:
            mse = calculate_mse(original, filtered_image)
            psnr = calculate_psnr(mse)

            # Apply edge detection:
            edges_original = apply_edge_detection(original)
            edges_filtered = apply_edge_detection(filtered_image)

            # حساب Pratt's FOM
            fom_score = calculate_fom(edges_original, edges_filtered)

            # Calculate elapsed time:
            elapsed_time = Time[k]
            
            
            '''
            # فتح الملف في وضع الإضافة ('a' لإضافة نص إلى نهاية الملف)
            with open('output.txt', 'a') as file:
                # كتابة نص إضافي إلى الملف
                file.write(f"\nEvaluating filters for {title} - {noise_type} Noise with {noise_level} level")
                file.write(f"\n{filter_name} - Kernel {k}x{k}:")
                file.write(f"\n  MSE: {mse:.2f}")
                file.write(f"\n  PSNR: {psnr:.2f} dB")
                file.write(f"\n  Computational Time: {elapsed_time:.4f} seconds\n")
            '''

            log_results(filter_name, k, noise_type, noise_level, mse, psnr, elapsed_time, fom_score)
            # Save the edge detection result to PDF file:
            save_evaluation_results(original, noisy_image, filtered_image, filter_name, k, mse, psnr, edges_original, edges_filtered, fom_score, pdf)
    
    ####################################################################################################################################
    #                               Build a Functions to Plot the filtered images & Save them in PDF file                              #
    ####################################################################################################################################

    def save_evaluation_results(original, noisy_image, filtered_image, filter_name, kernel_size, mse, psnr, edges_original, edges_filtered, fom_score, pdf):
        plt.figure(figsize=(15, 10))
        plt.suptitle(f'{filter_name} - Kernel Size: {kernel_size}x{kernel_size}\nMSE: {mse:.2f}, PSNR: {psnr:.2f} dB, FOM: {fom_score:.2f}', fontsize=14)

        # Plot the original image:
        plt.subplot(2, 3, 1)
        plt.imshow(original, cmap='gray')
        plt.title('Original Image')
        plt.axis('off')

        # Plot the noisy image:
        plt.subplot(2, 3, 2)
        plt.imshow(noisy_image, cmap='gray')
        plt.title('Noisy Image')
        plt.axis('off')

        # Plot the image after apply the filter on it:
        plt.subplot(2, 3, 3)
        plt.imshow(filtered_image, cmap='gray')
        plt.title(f'Filtered Image ({filter_name})')
        plt.axis('off')

        # Plot the edge for the original image:
        plt.subplot(2, 3, 4)
        plt.imshow(edges_original, cmap='gray')
        plt.title('Edges of Original Image')
        plt.axis('off')

        # Plot the edge for the filtered image:
        plt.subplot(2, 3, 5)
        plt.imshow(edges_filtered, cmap='gray')
        plt.title(f'Edges after {filter_name}')
        plt.axis('off')

        # Save the result to the PDF file:
        pdf.savefig()
        plt.close()
    
    '''
    --> The results are saved to a PDF file using the save_filtered_images function. 
    Each page in the PDF file contains the original image, blurred image and filtered images with different kernel size.
    '''

    def save_filtered_images(original, noisy_image, filter_name, filtered_images, kernel_sizes, title, noise_type, noise_level, pdf):
        plt.figure(figsize=(20, 10))
        plt.suptitle(f'{title} - {filter_name} - {noise_type} Noise with {noise_level} level', fontsize=16)

        # Plot the original image:
        plt.subplot(2, 4, 1)
        plt.imshow(original, cmap='gray')
        plt.title('Original')
        plt.axis('off')

        # Plot the image with noise:
        plt.subplot(2, 4, 2)
        plt.imshow(noisy_image, cmap='gray')
        plt.title(f'Noisy Image ({noise_level})')
        plt.axis('off')

        # Plot the images after applying the filter with different kernel sizes:
        for i, (filtered_image, k) in enumerate(zip(filtered_images, kernel_sizes)):
            plt.subplot(2, 4, i + 3)
            plt.imshow(filtered_image, cmap='gray')
            plt.title(f'Kernel {k}x{k}')
            plt.axis('off')

        # Save the figure in the pdf file:
        if plt.get_fignums():
            pdf.savefig()
            plt.close()
        else:
            print(f"Warning: No figure to save for {filter_name} - {noise_level}")

    ####################################################################################################################################
    #                                                     Apply the Filters to images                                                  #
    ####################################################################################################################################

    '''
    --> All filters are applied to each blurred image with different noise levels (low, medium, high) and the results are saved to a PDF file.
    '''
    def apply_all_filters_and_save(original, noisy_image, title, noise_type, noise_level, pdf, pdf2):

        # Apply simple filters:

        # 1- Apply Box Filter:
        box_filtered_images = [apply_box_filter(noisy_image, k, calculate_time[title][noise_type][noise_level]['Box']) for k in kernel_sizes]
        evaluate_filters(original, noisy_image, title, noise_type, noise_level, 'Box Filter', box_filtered_images, kernel_sizes, calculate_time[title][noise_type][noise_level]['Box'], pdf2)
        save_filtered_images(original, noisy_image, 'Box Filter', box_filtered_images, kernel_sizes, title, noise_type, noise_level, pdf)

        # 2- Apply Gaussian Filter:
        gaussian_filtered_images = [apply_gaussian_filter(noisy_image, k, calculate_time[title][noise_type][noise_level]['Gaussian']) for k in kernel_sizes]
        evaluate_filters(original, noisy_image, title, noise_type, noise_level, 'Gaussian Filter', gaussian_filtered_images, kernel_sizes, calculate_time[title][noise_type][noise_level]['Gaussian'], pdf2)
        save_filtered_images(original, noisy_image, 'Gaussian Filter', gaussian_filtered_images, kernel_sizes, title, noise_type, noise_level, pdf)

        # 3- Apply Median Filter:
        median_filtered_images = [apply_median_filter(noisy_image, k, calculate_time[title][noise_type][noise_level]['Median']) for k in kernel_sizes]
        evaluate_filters(original, noisy_image, title, noise_type, noise_level, 'Median Filter', median_filtered_images, kernel_sizes, calculate_time[title][noise_type][noise_level]['Median'], pdf2)
        save_filtered_images(original, noisy_image, 'Median Filter', median_filtered_images, kernel_sizes, title, noise_type, noise_level, pdf)

        # Apply advanced filters:

        # 1- Apply Adaptive Median Filter:
        adaptive_mean_filtered_images = [apply_adaptive_mean_filter(noisy_image, k, calculate_time[title][noise_type][noise_level]['Adaptive Mean']) for k in kernel_sizes]
        evaluate_filters(original, noisy_image, title, noise_type, noise_level, 'Adaptive Mean Filter', adaptive_mean_filtered_images, kernel_sizes, calculate_time[title][noise_type][noise_level]['Adaptive Mean'], pdf2)
        save_filtered_images(original, noisy_image, 'Adaptive Mean Filter', adaptive_mean_filtered_images, kernel_sizes, title, noise_type, noise_level, pdf)
        
        # 2- Apply Adaptive Median Filter:
        adaptive_median_filtered_images = [apply_adaptive_median_filter(noisy_image, k, calculate_time[title][noise_type][noise_level]['Adaptive Median']) for k in kernel_sizes]
        evaluate_filters(original, noisy_image, title, noise_type, noise_level, 'Adaptive Median Filter', adaptive_median_filtered_images, kernel_sizes, calculate_time[title][noise_type][noise_level]['Adaptive Median'], pdf2)
        save_filtered_images(original, noisy_image, 'Adaptive Median Filter', adaptive_median_filtered_images, kernel_sizes, title, noise_type, noise_level, pdf)

        # 3- Apply Bilateral Filter:
        bilateral_filtered_images = [apply_bilateral_filter(noisy_image, k, calculate_time[title][noise_type][noise_level]['Bilateral']) for k in kernel_sizes]
        evaluate_filters(original, noisy_image, title, noise_type, noise_level, 'Bilateral Filter', bilateral_filtered_images, kernel_sizes, calculate_time[title][noise_type][noise_level]['Bilateral'], pdf2)
        save_filtered_images(original, noisy_image, 'Bilateral Filter', bilateral_filtered_images, kernel_sizes, title, noise_type, noise_level, pdf)
    
    ####################################################################################################################################
    #                                                            Run the code                                                          #
    ####################################################################################################################################

    # Create PDF and save figures:
    filtered_pdf_file_name = 'filtered_images.pdf'
    edge_detection_pdf_file_name = 'edge_detection_images.pdf'

    with PdfPages(filtered_pdf_file_name) as pdf:
        with PdfPages(edge_detection_pdf_file_name) as pdf2:
            # Apply the filteron all the noisy images:
            for noisy_image, title, noise_type, noise_level, original in images:
                apply_all_filters_and_save(original, noisy_image, title, noise_type, noise_level, pdf, pdf2)

    print("Filtered results have been saved to '{}' and edge detection result have been saved in '{}'.".format(filtered_pdf_file_name, edge_detection_pdf_file_name))

    results_df = pd.DataFrame(results_data)
    results_df.to_csv("results.csv", index=False)
    print("Results saved to results.csv")

else:
    print("One or more images failed to load. Please check the image paths and try again.")