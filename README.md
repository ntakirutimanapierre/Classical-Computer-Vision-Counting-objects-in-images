# README

## Introduction
This repository contains Python scripts demonstrating various image processing techniques using OpenCV and Matplotlib libraries.

## Requirements
- Python 3.x
- OpenCV (`cv2`)
- Matplotlib (`matplotlib`)
- NumPy (`numpy`)

## Code Overview

### 1. Count Objects in an Image

#### Script: `count_objects`

- **Functionality**: Counts distinct objects in a given image using contour detection.
- **Usage**: Pass the path to an image to `count_objects(image_path)`.
- **Dependencies**: Uses OpenCV for image loading, conversion, and contour detection.

### 2. Image Compression using FFT

#### Script: `compress_image`

- **Functionality**: Compresses an image using Fast Fourier Transform (FFT).
- **Usage**: Provide the image path and compression ratio to `compress_image(image_path, compress_ratio)`.
- **Dependencies**: Relies on OpenCV for image operations and NumPy for FFT.

### 3. Image Blending

#### Script: `compress_image` (overloaded)

- **Functionality**: Blends two images using weighted addition.
- **Usage**: Supply two images and their respective weights to `compress_image(img1, weight_1, img2, weight_2)`.
- **Dependencies**: Requires OpenCV for image manipulation.

### 4. Data Augmentation

#### Script: `data_augmentation`

- **Functionality**: Performs various data augmentation techniques on an image.
- **Usage**: Pass an image and augmentation type to `data_augmentation(img, type)`.
- **Supported Augmentations**: Resize (nearest neighbor and cubic spline), vertical and horizontal flips, blur, rotation, and shear.

## Example Usage

### Example 1: Count Objects
```python
path1 = 'data/assignment1A.png'
path2 = 'data/assignment1B.png'

img1 = count_objects(path1)
print(f'Objects in {path1}: {img1}')

img2 = count_objects(path2)
print(f'Objects in {path2}: {img2}')

```



These scripts demonstrate fundamental image processing operations such as object counting, image compression, blending, and data augmentation using OpenCV and NumPy libraries. Each script can be adapted and expanded for specific use cases in image analysis and computer vision applications.
