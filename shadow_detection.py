# shadow_detection.py
import cv2
from google.colab.patches import cv2_imshow
import numpy as np
import matplotlib.pyplot as plt

def display_image(title, image):
    print(title + ':')
    cv2_imshow(image)

def main():
    # Step 1: Read input image
    input_image = cv2.imread('input/path/to/here.jpg')

    # Step 2: Apply Gaussian filter
    blurred_image = cv2.GaussianBlur(input_image, (3, 3), 0)

    # Step 3: Apply mean shift algorithm
    meanshift_image = cv2.pyrMeanShiftFiltering(blurred_image, 5, 10)

    # Step 4: Apply morphology operations (optional)
    kernel = np.ones((3, 3), np.uint8)
    morphology_image = cv2.morphologyEx(meanshift_image, cv2.MORPH_CLOSE, kernel)

    # Step 5: Convert to grayscale
    gray_image = cv2.cvtColor(morphology_image, cv2.COLOR_BGR2GRAY)

    # Step 6: Create mask (Background, Foreground, Mask Boundaries)
    _, thresholded_image = cv2.threshold(gray_image, 120, 255, cv2.THRESH_BINARY_INV)

    # Display intermediate results
    display_image('Original Image', input_image)
    display_image('Blurred Image', blurred_image)
    display_image('After Mean Shift', meanshift_image)
    display_image('Grayscale Image', gray_image)
    display_image('Thresholded Image', thresholded_image)

    # Step 7: Create mask using Canny edge detection
    edges = cv2.Canny(thresholded_image, 30, 100)

    # Display edge detection results
    display_image('Edges', edges)

    # Step 8: Dilate the edges to make the lines thicker
    dilated_edges = cv2.dilate(edges, None, iterations=2)

    # Display dilated edges
    display_image('Dilated Edges', dilated_edges)

    # Step 9: Draw a red line along the shadow boundaries
    result_image = input_image.copy()
    result_image[dilated_edges != 0] = [0, 0, 255]  # Set the red channel to 255 where edges are detected

    # Display result image
    display_image('Result Image', result_image)

    # Step 10: Find contours of the edges
    contours, _ = cv2.findContours(dilated_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Step 11: Extract boundary points
    boundary_points = [point[0] for contour in contours for point in contour]

    # Step 12: Extract x and y coordinates
    x_coordinates, y_coordinates = zip(*boundary_points)

    # Step 13: Define xmin, xmax, ymin, ymax
    xmin, xmax, ymin, ymax = min(x_coordinates), max(x_coordinates), min(y_coordinates), max(y_coordinates)

    # Step 14: Apply decrease contrast within the bounding box
    area = result_image[ymin:ymax, xmin:xmax]
    area = cv2.bilateralFilter(area, 2, 20, 50)
    # area = cv2.pyrMeanShiftFiltering(area, 5, 10)
    area = cv2.medianBlur(area, 5, 10)
    area = cv2.GaussianBlur(area, (3, 3), 3)

    # Step 15: Replace the area in the original image
    result_image[ymin:ymax, xmin:xmax] = area

    # Display the final result
    display_image('Final Result', result_image)

    # Save the result image
    cv2.imwrite('output/path/to/here.jpg', result_image)

    # Load the image for histogram analysis
    image = cv2.imread('output/path/to/here/.jpg', cv2.IMREAD_GRAYSCALE)

    # Calculate histogram
    histogram, bins = np.histogram(image.flatten(), bins=256, range=[0, 256])

    # Plot the histogram
    plt.figure(figsize=(10, 5))
    plt.plot(histogram, color='black')
    plt.xlabel('Pixel Value')
    plt.ylabel('Frequency')
    plt.title('Result Image Histogram')
    plt.show()

    # Load the original image for histogram analysis
    image2 = cv2.imread('input/path/to/here.jpg', cv2.IMREAD_GRAYSCALE)

    # Calculate histogram
    histogram2, bins2 = np.histogram(image2.flatten(), bins=256, range=[0, 256])

    # Plot the histogram
    plt.figure(figsize=(10, 5))
    plt.plot(histogram2, color='black')
    plt.xlabel('Pixel Value')
    plt.ylabel('Frequency')
    plt.title('Original Image Histogram')
    plt.show()

if __name__ == "__main__":
    main()
