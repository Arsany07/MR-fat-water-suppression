import cv2
import numpy as np
import pydicom
import matplotlib.pyplot as plt
import os

############################################## Read DICOM files ##############################################
def read_dicom_files(in_phase_file, out_phase_file):
    """
    Read DICOM files and extract the pixel array for in-phase and out-phase images.

    Parameters:
    in_phase_file (str): The file path of the in-phase DICOM image.
    out_phase_file (str): The file path of the out-phase DICOM image.

    Returns:
    tuple: A tuple containing the pixel array of the in-phase image and the out-phase image.
           If any error occurs during reading, returns (None, None).
    """
    try:
        dicom_in_phase_image = pydicom.dcmread(in_phase_file)
        in_phase_image = dicom_in_phase_image.pixel_array
    except (FileNotFoundError, pydicom.errors.InvalidDicomError):
        print(f"Error reading in-phase DICOM file: {in_phase_file}")
        return None, None

    try:
        dicom_out_phase_image = pydicom.dcmread(out_phase_file)
        out_phase_image = dicom_out_phase_image.pixel_array
    except (FileNotFoundError, pydicom.errors.InvalidDicomError):
        print(f"Error reading out-phase DICOM file: {out_phase_file}")
        return None, None

    return in_phase_image, out_phase_image


############################################### Clean the data ###############################################
def min_max_normalization(image):
    """
    Normalize the pixel values of an image to the range [0, 255] using min-max normalization.

    Parameters:
    image (numpy.ndarray): The input image to be normalized.

    Returns:
    numpy.ndarray: The normalized image with pixel values in the range [0, 255].
    """
    minimum_value = np.min(image)
    maximum_value = np.max(image)
    normalized_image = (image - minimum_value) / (maximum_value - minimum_value)
    normalized_image = np.interp(normalized_image, (normalized_image.min(), normalized_image.max()), (0, 255))
    return normalized_image.astype(np.uint8)


############################################### DIXON ALGORITHM ###############################################
def dixon_algorithm(in_phase_image, out_phase_image):
    """
    Apply the Dixon algorithm to separate fat and water components from the input in-phase and out-phase images.

    Parameters:
    in_phase_image (numpy.ndarray): The input in-phase image to be processed.
    out_phase_image (numpy.ndarray): The input out-phase image to be processed.

    Returns:
    tuple: A tuple containing the water image and the fat image after applying the Dixon algorithm.
    """
    water_image = cv2.add(in_phase_image, out_phase_image) / 2
    fat_image = cv2.subtract(in_phase_image, out_phase_image) / 2
    return water_image, fat_image


############################################ Enhancing the outputs ############################################
def enhance_images(water_image, fat_image):
    """
    Enhance the water and fat images by normalizing the pixel values and applying histogram equalization.
    
    Parameters:
        water_image (numpy.ndarray): The input water image to be enhanced.
        fat_image (numpy.ndarray): The input fat image to be enhanced.
    
    Returns:
        tuple: A tuple containing the enhanced water image and the enhanced fat image.
    """
    normalized_water_image = min_max_normalization(water_image)
    normalized_fat_image = min_max_normalization(fat_image)

    equalized_water_image = cv2.equalizeHist(normalized_water_image)
    equalized_fat_image = cv2.equalizeHist(normalized_fat_image)

    return equalized_water_image, equalized_fat_image

################################################### Display ###################################################
def display_images(in_phase_image, out_phase_image, water_image, fat_image, gauss_water_image, gauss_fat_image):
    """
    Display the input in-phase and out-phase images, the water and fat images, and the water and fat images after applying a Gaussian filter.
    
    Parameters:
        in_phase_image (numpy.ndarray): The input in-phase image to be displayed.
        out_phase_image (numpy.ndarray): The input out-phase image to be displayed.
        water_image (numpy.ndarray): The water image to be displayed.
        fat_image (numpy.ndarray): The fat image to be displayed.
        gauss_water_image (numpy.ndarray): The water image after applying a Gaussian filter to be displayed.
        gauss_fat_image (numpy.ndarray): The fat image after applying a Gaussian filter to be displayed.
    """
    in_phase_image_rgb = cv2.cvtColor(in_phase_image, cv2.COLOR_BGR2RGB)
    out_phase_image_rgb = cv2.cvtColor(out_phase_image, cv2.COLOR_BGR2RGB)
    equalized_water_image_rgb = cv2.cvtColor(water_image, cv2.COLOR_BGR2RGB)
    equalized_fat_image_rgb = cv2.cvtColor(fat_image, cv2.COLOR_BGR2RGB)
    equalized_gauss_water_image_rgb = cv2.cvtColor(gauss_water_image, cv2.COLOR_BGR2RGB)
    equalized_gauss_fat_image_rgb = cv2.cvtColor(gauss_fat_image, cv2.COLOR_BGR2RGB)

    fig, axes = plt.subplots(3, 2, figsize=(12, 12))

    axes[0, 0].imshow(in_phase_image_rgb, cmap='gray')
    axes[0, 0].set_title("In Phase Image")
    axes[0, 0].axis("off")

    axes[0, 1].imshow(out_phase_image_rgb, cmap='gray')
    axes[0, 1].set_title("Out Phase Image")
    axes[0, 1].axis("off")

    axes[1, 0].imshow(equalized_water_image_rgb, cmap='gray')
    axes[1, 0].set_title("Water Image")
    axes[1, 0].axis("off")

    axes[1, 1].imshow(equalized_fat_image_rgb, cmap='gray')
    axes[1, 1].set_title("Fat Image")
    axes[1, 1].axis("off")

    axes[2, 0].imshow(equalized_gauss_water_image_rgb, cmap='gray')
    axes[2, 0].set_title("Water Image (Gaussian filter)")
    axes[2, 0].axis("off")

    axes[2, 1].imshow(equalized_gauss_fat_image_rgb, cmap='gray')
    axes[2, 1].set_title("Fat Image (Gaussian filter)")
    axes[2, 1].axis("off")

    plt.suptitle("Fat-Water Suppression using Dixon Technique", fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

################################################### Main Function ###################################################
def main():

    BASE_DIR = "transverse_heart (complete data)/CPTAC-CCRCC/C3N-02262/08-17-2000-NA-MR JB-70639/5.000000-Ax T1 DualEcho outin Phase-97241"
    
    # in phase -> even numbers, Out of Phases -> odd numbers
    in_phase_file = os.path.join(BASE_DIR, "1-06.dcm")
    out_phase_file = os.path.join(BASE_DIR, "1-05.dcm")

    in_phase_image, out_phase_image = read_dicom_files(in_phase_file, out_phase_file)
    if in_phase_image is None or out_phase_image is None:
        return

    in_phase_image = min_max_normalization(in_phase_image)
    out_phase_image = min_max_normalization(out_phase_image)

    gaussian_in_phase_image = cv2.GaussianBlur(in_phase_image, (5, 5), 0)
    gaussian_out_phase_image = cv2.GaussianBlur(out_phase_image, (5, 5), 0)

    water_image, fat_image = dixon_algorithm(in_phase_image, out_phase_image)
    gauss_water_image, gauss_fat_image = dixon_algorithm(gaussian_in_phase_image, gaussian_out_phase_image)

    equalized_water_image, equalized_fat_image = enhance_images(water_image, fat_image)
    equalized_gauss_water_image, equalized_gauss_fat_image = enhance_images(gauss_water_image, gauss_fat_image)

    display_images(in_phase_image, out_phase_image, equalized_water_image, equalized_fat_image, equalized_gauss_water_image, equalized_gauss_fat_image)

if __name__ == "__main__":
    main()