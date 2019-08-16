import numpy as np
import cv2 as cv


# find the projector FOV mask
def thresh(im_in):
    # threshold im_diff with Otsu's method
    if im_in.ndim == 3:
        im_in = cv.cvtColor(im_in, cv.COLOR_BGR2GRAY)
    if im_in.dtype == 'float32':
        im_in = np.uint8(im_in * 255)
    _, im_mask = cv.threshold(cv.GaussianBlur(im_in, (5, 5), 0), 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
    im_mask = im_mask > 0

    # find the largest contour by area then convert it to convex hull
    im_contours, contours, hierarchy = cv.findContours(np.uint8(im_mask), cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    hulls = cv.convexHull(max(contours, key=cv.contourArea))
    im_mask = cv.fillConvexPoly(np.zeros_like(im_mask, dtype=np.uint8), hulls, True) > 0

    # also calculate the bounding box
    bbox = cv.boundingRect(max(contours, key=cv.contourArea))
    corners = [[bbox[0], bbox[1]], [bbox[0] + bbox[2], bbox[1]], [bbox[0] + bbox[2], bbox[1] + bbox[3]], [bbox[0], bbox[1] + bbox[3]]]

    # normalize to (-1, 1) following pytorch grid_sample coordinate system
    h = im_in.shape[0]
    w = im_in.shape[1]

    for pt in corners:
        pt[0] = 2 * (pt[0] / w) - 1
        pt[1] = 2 * (pt[1] / h) - 1

    return im_mask, corners
