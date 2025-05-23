# Fat-Water Separation in MRI Imaging

## Overview

This project implements the Dixon technique to separate fat and water signals in MRI images using Python. The method leverages the differences in precessional frequencies between fat and water protons, enabling improved diagnostic insights such as organ composition analysis and detection of fatty infiltration.

## Objectives

- Understand and implement the Dixon technique for fat-water separation in MRI.
- Process in-phase and out-of-phase MRI DICOM images.
- Apply image enhancement techniques to improve visualization.

## Methodology

1. **Read DICOM Files**  
   Used the `pydicom` library to load in-phase and out-of-phase MRI images.

2. **Data Normalization**  
   Applied min-max normalization to scale image intensities and Gaussian filtering for noise reduction.

3. **Dixon Algorithm**  
   - **Water Image**: `(In-phase + Out-of-phase) / 2`
   - **Fat Image**: `(In-phase - Out-of-phase) / 2`

4. **Image Enhancement**  
   Further normalized and applied histogram equalization to enhance contrast.

5. **Visualization**  
   Displayed original and processed images in a 3x2 subplot for comparative analysis.

## Tools & Libraries

- Python  
- pydicom  
- NumPy  
- OpenCV  
- Matplotlib  

## Sample Results
Visualization includes:
- In-phase and Out-of-phase MRI images
- Resulting Water and Fat images (with and without filtering)
- ![Results](https://github.com/Arsany07/MR-fat-water-suppression/blob/mainResults.png))

## References

1. Dixon, W. T. (1984). Simple proton spectroscopic imaging. *Radiology*, 153(1), 189–194.  
2. Reeder, S. B. et al. (2005). IDEAL: Application with fast spin-echo imaging. *Magnetic Resonance in Medicine*, 54(3), 636–644.
