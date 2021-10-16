import numpy as np
import cv2

# Detect boundaries
# Ideal brown: [19, 69, 139]
# HSV: [25, 76, 31]
# brown_min_bgr = np.array([0, 0, 0], dtype=np.uint8)
# brown_max_bgr = np.array([40, 90, 160], dtype=np.uint8)
# 10, 50, 20, 20, 255, 200
# brown_min_hsv = np.array([10, 30, 20], dtype=np.uint8)
# brown_max_hsv = np.array([20, 255, 200], dtype=np.uint8)
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
def visualize_boundary(frame, c):
    viz = np.copy(frame)

    # Draw contours on image
    cv2.drawContours(viz, [c], 0, (0, 255, 0), 3)
    return viz


def main():

    cap = cv2.VideoCapture(2)

    while True:
        _, frame = cap.read()
        boundary_viz = np.zeros(shape=frame.shape)

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        ## Detect Boundaries ##

        # Find the colors within the boundaries and mask
        # mask = cv2.inRange(frame, brown_min_bgr, brown_max_bgr)
        boundary_mask = cv2.inRange(hsv, blue_min_hsv, blue_max_hsv)
        bgr_mask = cv2.cvtColor(boundary_mask, cv2.COLOR_GRAY2BGR)

        # Create kernels for dilution and erosion operations; larger ksize means larger pixel neighborhood where the
        # operation is taking place
        se1 = cv2.getStructuringElement(cv2.MORPH_RECT, ksize=(2, 2))
        se2 = cv2.getStructuringElement(cv2.MORPH_RECT, ksize=(20, 20))

        # Perform "opening_frame." This is erosion followed by dilation, which reduces noise. The ksize is pretty small,
        # otherwise the white in the marker is eliminated
        # opening_frame = cv2.morphologyEx(boundary_mask, cv2.MORPH_OPEN, se1)

        # Perform "closing." This is dilution followed by erosion, which fills in black gaps within the marker. This is
        # necessary if the lightness threshold is not able to get the entire marker at lower altitudes
        processed_frame = cv2.morphologyEx(boundary_mask, cv2.MORPH_CLOSE, se2)

        # Find contours
        contours, _ = cv2.findContours(
            processed_frame, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE
        )

        # Find corners of the contour
        if len(contours) >= 1:

            # Take contour w/ max area; the marker will always be the largest contour in the image
            c = max(contours, key=cv2.contourArea)

            # Approximate the contour
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.03 * peri, True)

            # Keep going if the contour is a square
            if is_rectangle(approx):
                c = approx

                # Visualize
                boundary_viz = visualize_boundary(frame, c)

        viz = cv2.cvtColor(processed_frame, cv2.COLOR_GRAY2BGR)
        cv2.imshow("Original Image", np.hstack([frame, bgr_mask]))
        cv2.imshow("Field Detection", boundary_viz)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
