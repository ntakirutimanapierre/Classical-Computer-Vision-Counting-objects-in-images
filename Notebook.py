import cv2
import matplotlib.pyplot as plt
import numpy as np

## Sample images  in Data directory
def count_objects(image_path):
    # Load the image using opencv
    img = cv2.imread(image_path)

    # Convert the image to grayscale for easy manipulation
    gray_scale = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply GaussianBlur to reduce noise and help contour detection
    blur_image = cv2.GaussianBlur(gray_scale, (5, 5),0)

    

    # Use Canny edge detection we could also use sobel
    # canny_image = cv2.Canny(blur_image, 90,)
    canny_image = cv2.Canny(blur_image, 78, 90)

    # Find contours in the edges, this can help to detect the edges 
    contours, hierarchy = cv2.findContours(canny_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Count the number of objects which should just be the number of contours you have i
    number_of_contours = len(contours)
    return number_of_contours #the number of objects



path1 = 'data/assignment1A.png'
path2 = 'data/assignment1B.png'
img1 = count_objects(path1)
print(f'pntakiru:{img1}')

img2 = count_objects(path2)
print(f'pntakiru:{img2}')


# 2222222222222222222222222222222222222
def compress_image(image_path, compress_ratio):
    # Load image using imread
    image = cv2.imread(image_path)

    # Convert image to grayscale
    grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Use numpy to get the fast Fourier transform of the grayscale image
    computed_fourier = np.fft.fft2(grayscale_image)
    
    # Shift the zero frequency component to the center
    fshift = np.fft.fftshift(computed_fourier)

    # Flatten the frequencies and sort the absolute values
    Flatten_sorted = np.sort(np.abs(fshift).flatten())

    # Determine the threshold to keep using thresh = f_sorted[int(np.floor((1 - compress_ratio) * len(f_sorted)))]
    threshold = Flatten_sorted[int(np.floor((1 - compress_ratio) * len(Flatten_sorted)))]

    # Create a mask to help filter and Filter the Fourier using the mask
    mask = np.abs(fshift) > threshold
    filtered_fourier = fshift * mask

    # Apply inverse Fourier transform to get a compressed version of the image
    compressed_image = np.fft.ifft2(np.fft.ifftshift(filtered_fourier)).real

    # Convert the compressed image back to uint8 for proper display
    compressed_image = np.uint8(compressed_image)

    return compressed_image

image_path = 'Data/question2_high_frequency.jpg'
image_path2 = 'Data/question2_image1.jpg'


#0.1, 0.01 and 0.002
ratios = [0.1, 0.01, 0.002]
fig, ax = plt.subplots(nrows=1, ncols=4, figsize=(15, 7))

im = cv2.imread(image_path)
# im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY, cmap = 'gray')
gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
ax[0].imshow(gray, cmap = 'gray')
ax[0].set_title('Original Image')

for index, ratio in enumerate(ratios):
    img = compress_image(image_path = image_path, compress_ratio = ratio)
    ax[index+1].imshow(img, cmap='gray')
    ax[index+1].set_title(f'Compressed Image: ratio {ratio}')
plt.tight_layout()
plt.show()


fig, ax = plt.subplots(nrows=1, ncols=4, figsize=(15, 7))

im = cv2.imread(image_path2)
# im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY, cmap = 'gray')
gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
ax[0].imshow(gray, cmap = 'gray')
ax[0].set_title('Original Image')

for index, ratio in enumerate(ratios):
    img = compress_image(image_path = image_path2, compress_ratio = ratio)
    ax[index+1].imshow(img, cmap='gray')
    ax[index+1].set_title(f'Compressed Image: ratio {ratio}')
plt.tight_layout()
plt.show()




#3333333333333333333333333333333333333
##  Use your own image to test
def compress_image(img1, weight_1, img2, weight_2):
    # Use opencv addWeighted to blend both images
    img1 = cv2.resize(img1, (img2.shape[1], img2.shape[0])) 
    blended = cv2.addWeighted(img1, weight_1, img2, weight_2, 0)
    # return blended image
    return blended

file1 = 'Data/question2_high_frequency.jpg'
file2 = 'Data/question2_image1.jpg'
img1 = cv2.imread(file1)
img2 = cv2.imread(file2) 



weight_1 = 0.5
weight_2 = 0.7
blended = compress_image(img1 = img1, weight_1 = weight_1, img2 = img2, weight_2 = weight_2)



fig, ax = plt.subplots(nrows=1, ncols=3, figsize = (15, 5))
ax[2].imshow(blended)
ax[2].set_title(f'Blended Image: Weights {weight_1, weight_2}')


ax[1].imshow(img1)
ax[1].set_title('Original Image2')

ax[0].imshow(img2)
ax[0].set_title('Original Image1')
plt.show()


#444444444444444444444444444444444444
## Use your own images to test.
def data_augmentation(img, type):
    if type=="resize1":
        # resize the  image to 224 x 224 use nearest neighbor 
        img = cv2.resize(src=img, dsize=(224, 224), fx = 0.75, fy = 0.75, interpolation=cv2.INTER_NEAREST)
    elif type=="resize2":
        # resize the  image to 224 x 224 use cubic spline interpolation
        img = cv2.resize(src=img, dsize=(224, 224), fx = 0.75, fy = 0.75, interpolation=cv2.INTER_CUBIC)
    elif type=="vertical_flip":
        # flip the image vertically
        img = cv2.flip(src = img, flipCode = 0)
    elif type=="horizontal_flip":
        # flip the image horizontally
        img = cv2.flip(img, flipCode = 1)
    elif type=="blur_noise":
        # flip the image horizontally
        img = cv2.blur(src=img, ksize=(5, 5))
    elif type=="rotation":
        # use your own parameters
        img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    elif type=="shear on y-axis":
        # use your own parameters
        M = np.float32([[1, 0.5, 0],
             	[0, 1  , 0],
            	[0, 0  , 1]])
        
        img = cv2.warpPerspective(img, M=M, dsize=img.shape)
    elif type=="shear on x-axis":
        # use your own parameters
        M = np.float32([[1, 0, 0],
             	[0.5, 1  , 0],
            	[0, 0  , 1]])
        img = cv2.warpPerspective(img, M=M, dsize=img.shape)
    return img


fig, ax = plt.subplots(2, 4, figsize=(20, 10))

gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
 
# Show the original image
ax[0,0].imshow(gray)
ax[0,0].set_title('Original Image')


# show the resized image
hhh = data_augmentation(gray, "resize1")
ax[0,1].imshow(hhh)
ax[0,1].set_title('Resize Image: nearest neighbor')


# show the resized image
hhh = data_augmentation(gray, "resize2")
ax[0,2].imshow(hhh)
ax[0,2].set_title('Resize Image: cubic spline interpolation')

# show the vertical flipped image
hhh = data_augmentation(gray, "vertical_flip")
ax[0,3].imshow(hhh)
ax[0,3].set_title('Vertical Flip')


hhh = data_augmentation(gray, "horizontal_flip")
ax[1,0].imshow(hhh)
ax[1,0].set_title('Horizontal Flip')

hhh = data_augmentation(gray, "blur_noise")
ax[1,1].imshow(hhh)
ax[1,1].set_title('Blur Image')


hhh = data_augmentation(gray, "rotation")
ax[1,2].imshow(hhh)
ax[1,2].set_title('Rotation')

hhh = data_augmentation(gray, "shear on y-axis")
ax[1,3].imshow(hhh)
ax[1,3].set_title("shear on y-axis")
plt.show()

hhh = data_augmentation(gray, "shear on x-axis")
plt.imshow(hhh)
plt.title("shear on x-axis")
plt.show()