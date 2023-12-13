# shadow_detection.py
import cv2
import numpy as np

def main():
    # Step 1: Read input image
    input_image = cv2.imread('path/to/image/here/.jpg')

    # Step 2: Apply Gaussian filter
    blurred_image = cv2.GaussianBlur(input_image, (3, 3), 0)

    # Step 3: Apply mean shift algorithm
    # Set the spatial and color radius based on your image characteristics
    meanshift_image = cv2.pyrMeanShiftFiltering(blurred_image, 5, 10)

    # Step 4: Apply morphology operations (optional)
    # This step is optional and may depend on your specific case
    kernel = np.ones((3, 3), np.uint8)
    morphology_image = cv2.morphologyEx(meanshift_image, cv2.MORPH_CLOSE, kernel)

    # Step 5: Convert to grayscale
    gray_image = cv2.cvtColor(morphology_image, cv2.COLOR_BGR2GRAY)

    # Step 6: Create mask (Background, Foreground, Mask Boundaries)
    _, thresholded_image = cv2.threshold(gray_image, 120, 255, cv2.THRESH_BINARY_INV)

    # Step 7: Create mask using Canny edge detection
    edges = cv2.Canny(thresholded_image, 30, 100)

    # Step 8: Dilate the edges to make the lines thicker
    dilated_edges = cv2.dilate(edges, None, iterations=2)

    # Step 9: Draw a red line along the shadow boundaries
    result_image = input_image.copy()
    result_image[dilated_edges != 0] = [0, 0, 255]  # Set the red channel to 255 where edges are detected

    # Step 10: Save and show images
    cv2.imwrite('input_image.jpg', input_image)
    cv2.imwrite('blurred_image.jpg', blurred_image)
    cv2.imwrite('meanshift_image.jpg', meanshift_image)
    cv2.imwrite('morphology_image.jpg', morphology_image)
    cv2.imwrite('gray_image.jpg', gray_image)
    cv2.imwrite('thresholded_image.jpg', thresholded_image)
    cv2.imwrite('edges.jpg', edges)
    cv2.imwrite('dilated_edges.jpg', dilated_edges)
    cv2.imwrite('result_image.jpg', result_image)

    # Display images using cv2.imshow (uncomment if running locally)
    # cv2.imshow('Original Image', input_image)
    # cv2.imshow('Blurred Image', blurred_image)
    # cv2.imshow('After Mean Shift', meanshift_image)
    # cv2.imshow('Grayscale Image', gray_image)
    # cv2.imshow('Thresholded Image', thresholded_image)
    # cv2.imshow('Edges', edges)
    # cv2.imshow('Dilated Edges', dilated_edges)
    # cv2.imshow('Result Image', result_image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
