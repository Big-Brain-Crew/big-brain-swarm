import numpy as np
import cv2

# Blue color range for boundaries
blue_min_hsv = np.array([95, 160, 60], dtype=np.uint8)
blue_max_hsv = np.array([125, 255, 255], dtype=np.uint8)

# Checks if the contour approximation is a rectangle.
def is_rectangle(approx):

    # print(len(approx))
    if len(approx) == 4:

        # compute the bounding box of the contour and use the
        # bounding box to compute the aspect ratio
        (x, y, w, h) = cv2.boundingRect(approx)
        ar = w / float(h)

        # a rectangle will have an aspect ratio that is within this range
        return True if 0.6 <= ar <= 2.0 else False
    return False


# Draw the marker corners and pose info on the image for debugging.
def visualize_boundary(frame, corners, c):
    viz = np.copy(frame)

    for corner in corners:
        x,y = corner.ravel()
        cv2.circle(viz,(int(x),int(y)),8,(255,120,255),-1)

    # Draw contours on image
    cv2.drawContours(viz, [c], 0, (0, 255, 0), 3)
    return viz


def order_points(pts):
    # initialzie a list of coordinates that will be ordered
    # such that the first entry in the list is the top-left,
    # the second entry is the top-right, the third is the
    # bottom-right, and the fourth is the bottom-left
    rect = np.zeros((4, 2), dtype = "float32")
    # the top-left point will have the smallest sum, whereas
    # the bottom-right point will have the largest sum
    s = pts.sum(axis = 1)
    #   pdb;pdb.set_trace()
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    # now, compute the difference between the points, the
    # top-right point will have the smallest difference,
    # whereas the bottom-left will have the largest difference
    diff = np.diff(pts, axis = 1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    # return the ordered coordinates
    return rect


def four_point_transform(image, pts):
	# obtain a consistent order of the points and unpack them
	# individually
	rect = order_points(pts)
	(tl, tr, br, bl) = rect
	# compute the width of the new image, which will be the
	# maximum distance between bottom-right and bottom-left
	# x-coordiates or the top-right and top-left x-coordinates
	widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
	widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
	maxWidth = max(int(widthA), int(widthB))
	# compute the height of the new image, which will be the
	# maximum distance between the top-right and bottom-right
	# y-coordinates or the top-left and bottom-left y-coordinates
	heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
	heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
	maxHeight = max(int(heightA), int(heightB))
	# now that we have the dimensions of the new image, construct
	# the set of destination points to obtain a "birds eye view",
	# (i.e. top-down view) of the image, again specifying points
	# in the top-left, top-right, bottom-right, and bottom-left
	# order
	dst = np.array([
		[0, 0],
		[maxWidth - 1, 0],
		[maxWidth - 1, maxHeight - 1],
		[0, maxHeight - 1]], dtype = "float32")
	# compute the perspective transform matrix and then apply it
	M = cv2.getPerspectiveTransform(rect, dst)
	warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
	# return the warped image
	return warped


def main():

    cap = cv2.VideoCapture(2)

    while True:
        _, frame = cap.read()

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        ## Detect Boundaries ##

        # Find the colors within the boundaries and mask
        # mask = cv2.inRange(frame, brown_min_bgr, brown_max_bgr)
        boundary_mask = cv2.inRange(hsv, blue_min_hsv, blue_max_hsv)
        bgr_mask = cv2.cvtColor(boundary_mask, cv2.COLOR_GRAY2BGR)

        # Create kernels for dilution and erosion operations; larger ksize means larger pixel neighborhood where the
        # operation is taking place
        se1 = cv2.getStructuringElement(cv2.MORPH_RECT, ksize=(20, 20))

        # Perform "closing." This is dilution followed by erosion, which fills in black gaps within the marker. This is
        # necessary if the lightness threshold is not able to get the entire marker at lower altitudes
        processed_frame = cv2.morphologyEx(boundary_mask, cv2.MORPH_CLOSE, se1)

        # Find contours
        contours, _ = cv2.findContours(processed_frame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        # Find corners of the contour
        field = np.zeros(shape=frame.shape)
        inverted_frame = np.zeros(shape=frame.shape)
        filled_in_border = np.zeros(shape=frame.shape)
        boundary_viz = np.zeros(shape=frame.shape)
        if len(contours) >= 1:

            # Take contour w/ max area; the marker will always be the largest contour in the image
            c = max(contours, key=cv2.contourArea)

            # Approximate the contour
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.03 * peri, True)

            if is_rectangle(approx):

                # rect = cv2.minAreaRect(approx)
                # box = cv2.boxPoints(rect)
                # box = np.int0(box)
                # filled_in_border = processed_frame.copy()
                # cv2.fillPoly(filled_in_border, [box], (255,255,255))
                
                corners = cv2.goodFeaturesToTrack(processed_frame, maxCorners=4, qualityLevel=0.5, minDistance=150).squeeze()
                print(len(corners))
                for corner in corners:
                    x,y = corner.ravel()
                    cv2.circle(processed_frame,(int(x),int(y)),8,(255, 255, 255),-1)
                
                field = four_point_transform(frame, corners)                
                                
                # y_corners = [int(p[0]) for p in corners]
                # x_corners = [int(p[1]) for p in corners]
                # y_min, y_max, x_min, x_max = np.min(y_corners), np.max(y_corners), np.min(x_corners), np.max(x_corners)

                # # Invert
                # inverted_frame = cv2.bitwise_not(processed_frame[x_min:x_max, y_min:y_max])

                # # Find contours
                # contours, _ = cv2.findContours(inverted_frame, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
                
                # if len(contours) >= 1:

                #     # Take contour w/ max area; the marker will always be the largest contour in the image
                #     c = max(contours, key=cv2.contourArea)

                #     # Approximate the contour
                #     peri = cv2.arcLength(c, True)
                #     approx = cv2.approxPolyDP(c, 0.03 * peri, True)

                #     if is_rectangle(approx):
                #         c = approx                

                #         rect = cv2.minAreaRect(c)
                #         box = cv2.boxPoints(rect)
                #         corners = np.int0(box)

                #         y_corners = [int(p[0]) for p in corners]
                #         x_corners = [int(p[1]) for p in corners]
                #         y_min, y_max, x_min, x_max = np.min(y_corners), np.max(y_corners), np.min(x_corners), np.max(x_corners)                        
                #         inverted_frame = inverted_frame[x_min:x_max, y_min:y_max]

                        # corners = cv2.goodFeaturesToTrack(inverted_frame, maxCorners=4, qualityLevel=0.5, minDistance=150).squeeze()
                        # for corner in corners:
                        #     x,y = corner.ravel()
                        #     cv2.circle(inverted_frame,(int(x),int(y)),8,(255,120,255),-1)
                
                        # field = four_point_transform(frame, corners)

                        # boundary_viz = visualize_boundary(frame[x_min:x_max, y_min:y_max], corners, c)

        
        cv2.imshow("Original Image", np.hstack([frame, bgr_mask]))
        cv2.imshow("Field", field)
        cv2.imshow("Field Detection", processed_frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
