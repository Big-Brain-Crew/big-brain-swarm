import numpy as np
import cv2
import pdb

debug = True

# Blue color range for boundaries
blue_min_hsv = np.array([95, 160, 60], dtype=np.uint8)
blue_max_hsv = np.array([125, 255, 255], dtype=np.uint8)

bot_color_ranges = {
    "red": [
        [np.array([0, 80, 180]), np.array([10, 255, 255])],
        [np.array([160, 80, 180]), np.array([179, 255, 255])],
    ],
    "green": [[np.array([34, 100, 120]), np.array([80, 255, 255])]],
}


class LowPassFilter:
    def __init__(self, current_scale=0.5):
        self.previous_value_ = None
        self.current_scale_ = current_scale
        self.first = True

    def update(self, value):
        if self.first:
            self.previous_value_ = value
            self.first = False
        else:
            self.previous_value_ = (
                self.previous_value_ * (1 - self.current_scale_) + value * self.current_scale_
            )
        return self.previous_value_


corner_lpf = LowPassFilter(0.1)
swarm_position_lpf = LowPassFilter(0.1)


def is_rectangle(approx):

    # print(len(approx))p
    if len(approx) == 4:

        # compute the bounding box of the contour and use the
        # bounding box to compute the aspect ratio
        (x, y, w, h) = cv2.boundingRect(approx)
        ar = w / float(h)

        # a rectangle will have an aspect ratio that is within this range
        return True if 0.6 <= ar <= 2.0 else False
    return False


def is_triangle(approx):

    if len(approx) == 3:
        return True

    return False


# Draw the marker corners and pose info on the image for debugging.
def visualize_boundary(frame, corners, c):
    viz = np.copy(frame)

    for corner in corners:
        x, y = corner.ravel()
        cv2.circle(viz, (int(x), int(y)), 8, (255, 120, 255), -1)

    # Draw contours on image
    cv2.drawContours(viz, [c], 0, (0, 255, 0), 3)
    return viz


def order_points(pts):

    # Initialize a list of coordinates that will be ordered
    # such that the first entry in the list is the top-left,
    # the second entry is the top-right, the third is the
    # bottom-right, and the fourth is the bottom-left
    rect = np.zeros((4, 2), dtype="float32")

    # The top-left point will have the smallest sum, whereas
    # the bottom-right point will have the largest sum
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    # Compute the difference between the points, the
    # top-right point will have the smallest difference,
    # whereas the bottom-left will have the largest difference
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    return rect


def four_point_transform(image, rect):
    (tl, tr, br, bl) = rect

    # Compute the width of the new image, which will be the
    # maximum distance between bottom-right and bottom-left
    # x-coordiates or the top-right and top-left x-coordinates
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    # Compute the height of the new image, which will be the
    # maximum distance between the top-right and bottom-right
    # y-coordinates or the top-left and bottom-left y-coordinates
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    # Now that we have the dimensions of the new image, construct
    # the set of destination points to obtain a "birds eye view",
    # (i.e. top-down view) of the image, again specifying points
    # in the top-left, top-right, bottom-right, and bottom-left
    # order
    dst = np.array(
        [[0, 0], [maxWidth - 1, 0], [maxWidth - 1, maxHeight - 1], [0, maxHeight - 1]],
        dtype="float32",
    )

    # Compute the perspective transform matrix and then apply it
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

    return warped


def find_field_frame(frame):
    ret = 0
    field = np.zeros(shape=frame.shape)

    # HSV space less sensitive to light variations when detecting color
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Find the colors within the boundaries and mask
    border_mask = cv2.inRange(hsv, blue_min_hsv, blue_max_hsv)

    # Create kernels for dilution and erosion operations; larger ksize means larger pixel neighborhood where the
    # operation is taking place
    se1 = cv2.getStructuringElement(cv2.MORPH_RECT, ksize=(5, 5))
    se2 = cv2.getStructuringElement(cv2.MORPH_RECT, ksize=(50, 50))  # used to be 25

    # Erosion followed by dilation, for rremoving noise
    # opening_frame = cv2.morphologyEx(border_mask, cv2.MORPH_OPEN, se1)

    # Dilution followed by erosion, which fills in black gaps.
    processed_frame = cv2.morphologyEx(border_mask, cv2.MORPH_CLOSE, se2)

    # Find contours
    contours, _ = cv2.findContours(processed_frame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    if len(contours) >= 1:

        # Take contour w/ max area; the border will always be the largest contour in the image
        c = max(contours, key=cv2.contourArea)

        # Approximate the contour
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.03 * peri, True)

        if is_rectangle(approx):

            # Find the corners of the field
            corners = cv2.goodFeaturesToTrack(
                processed_frame, maxCorners=4, qualityLevel=0.5, minDistance=150
            ).squeeze()

            if len(corners) == 4:
                try:

                    # Order the points for the LPF and perspective transform
                    corners = order_points(corners)

                    # Stabilize corners from noise
                    corners = corner_lpf.update(corners)

                    # Perspective transform field to be top-down
                    field = four_point_transform(frame, corners)

                    # Success
                    ret = 1
                except:
                    print("Couldn't find field - corners are invalid")

            else:
                print(f"Only {len(corners)} corners found")
                pass

    if debug:
        border_mask_bgr = cv2.cvtColor(border_mask, cv2.COLOR_GRAY2BGR)
        # opening_frame_bgr = cv2.cvtColor(opening_frame, cv2.COLOR_GRAY2BGR)
        processed_frame_bgr = cv2.cvtColor(processed_frame, cv2.COLOR_GRAY2BGR)
        if ret:
            for corner in corners:
                x, y = corner.ravel()
                cv2.circle(processed_frame_bgr, (int(x), int(y)), 8, (0, 0, 255), -1)
        cv2.imshow(
            "HSV, Border Range, Morphed", np.hstack([hsv, border_mask_bgr, processed_frame_bgr])
        )
        cv2.waitKey(1)

    return ret, field


bot_contour_area = 1200
bot_contour_area_tolerance = 40


def find_contour_with_most_similar_area(contours):
    most_similar_c = None
    most_similar_area = -1000
    for c in contours:
        area = cv2.contourArea(c)
        print("AREA", area)
        # print(area > bot_contour_area - bot_contour_area_tolerance)
        # print(area < bot_contour_area + bot_contour_area_tolerance)
        # print(abs(area- bot_contour_area))
        if (
            area > bot_contour_area - bot_contour_area_tolerance
            and area < bot_contour_area + bot_contour_area_tolerance
            and abs(area - bot_contour_area) < abs(most_similar_area - bot_contour_area)
        ):
            most_similar_area = area
            most_similar_c = c

    return most_similar_c


# Assumes a triangle contour
def find_contour_orientation(contour):
    p1, p2, p3 = contour[0].squeeze(), contour[1].squeeze(), contour[2].squeeze()

    p1_distances = [
        np.sqrt(np.square(p1[0] - p2[0]) + np.square(p1[1] - p2[1])),
        np.sqrt(np.square(p1[0] - p3[0]) + np.square(p1[1] - p3[1])),
    ]
    p2_distances = [
        np.sqrt(np.square(p2[0] - p1[0]) + np.square(p2[1] - p1[1])),
        np.sqrt(np.square(p2[0] - p3[0]) + np.square(p2[1] - p3[1])),
    ]
    p3_distances = [
        np.sqrt(np.square(p3[0] - p1[0]) + np.square(p3[1] - p1[1])),
        np.sqrt(np.square(p3[0] - p2[0]) + np.square(p3[1] - p2[1])),
    ]

    sums = [sum(p1_distances), sum(p2_distances), sum(p3_distances)]
    p_front_idx = sums.index(max(sums))
    p_front = contour[p_front_idx].squeeze()
    back_points = np.delete(contour, p_front_idx, 0).squeeze()

    p_short_line_middle = np.array(
        [
            np.average([back_points[0][0], back_points[1][0]]),
            np.average([back_points[0][1], back_points[1][1]]),
        ]
    ).astype(int)

    theta = np.arctan2(p_front[1] - p_short_line_middle[1], p_front[0] - p_short_line_middle[0]) * (
        180.0 / np.pi
    )

    return theta, p_front, p_short_line_middle


def find_swarm_position(f, swarm):
    """
    Check for all bot colors

    """
    frame = f.copy()
    swarm_position = []

    # Debug values
    ret = 0
    if debug:
        contours_to_draw = []
        points_to_draw = []
        debug_mask = np.zeros(shape=(frame.shape[0], frame.shape[1]), dtype=np.uint8)

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    for bot in swarm:
        bot_position = None
        if bot in bot_color_ranges:
            ranges = bot_color_ranges[bot]

            full_mask = np.zeros(shape=(hsv.shape[0], hsv.shape[1]), dtype=np.uint8)
            for range in ranges:
                full_mask += cv2.inRange(hsv, range[0], range[1])

            se1 = cv2.getStructuringElement(cv2.MORPH_RECT, ksize=(5, 5))
            se2 = cv2.getStructuringElement(cv2.MORPH_RECT, ksize=(3, 3))

            # Erosion followed by dilation, for rremoving noise
            full_mask = cv2.morphologyEx(full_mask, cv2.MORPH_OPEN, se1)

            # Dilution followed by erosion, which fills in black gaps.
            full_mask = cv2.morphologyEx(full_mask, cv2.MORPH_CLOSE, se2)

            if debug:
                debug_mask += full_mask

            # Find contours
            contours, _ = cv2.findContours(full_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if len(contours) >= 1:
                c = contours[0]

                # Choose contour with area closest to calibrated area
                # TODO: Add back in area check when they have the same area
                # bot_candidate = find_contour_with_most_similar_area(contours)
                bot_candidate = c
                if bot_candidate is not None:

                    # Approximate the contour
                    peri = cv2.arcLength(c, True)
                    approx = cv2.approxPolyDP(c, 0.1 * peri, True)

                    if is_triangle(approx):
                        c = approx

                        # Find position of bot center
                        M = cv2.moments(c)
                        c_x = int(M["m10"] / M["m00"])
                        c_y = int(M["m01"] / M["m00"])

                        # Find orientation relative to top left of field (0,0)
                        theta, p_front, p_short_line_middle = find_contour_orientation(c)

                        bot_position = np.array([int(c_x), int(c_y), int(theta)])

                        ret = 1
                        if debug:
                            contours_to_draw.append(c)
                            points_to_draw.append(p_front)
                            points_to_draw.append(p_short_line_middle)
                            points_to_draw.append([c_x, c_y])

        swarm_position.append(bot_position)

    if debug:
        debug_mask_bgr = cv2.cvtColor(debug_mask, cv2.COLOR_GRAY2BGR)

        if ret:
            for c in contours_to_draw:
                cv2.drawContours(debug_mask_bgr, [c], 0, (0, 255, 0), 3)
            for p in points_to_draw:
                cv2.circle(debug_mask_bgr, p, 4, (0, 0, 255), -1)

        cv2.imshow("Swarm Detection", np.hstack([frame, hsv, debug_mask_bgr]))
        cv2.waitKey(1)

    return swarm_position


def find_obstacles(f):
    """
    Obstacles are considered to be gray-black objects placed in the field.

    Return mask of obstacles (w/ dilation) and mask of original obstacles
    """

    frame = f.copy()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blurred, 60, 255, cv2.THRESH_BINARY_INV)

    # Add some thickness to each obstacle - this will help bots avoid clipping corners
    kernel = np.ones((20, 20), np.uint8)
    dilated = cv2.dilate(thresh, kernel, iterations=1)

    if debug:
        cv2.imshow("Obstacle Detection", np.hstack([gray, blurred, thresh, dilated]))
        cv2.waitKey(1)

    return dilated, thresh


def add_obstacles_to_occupancy_grid(occupancy_grid, obstacles):
    updated_grid = cv2.bitwise_or(occupancy_grid, obstacles)
    return updated_grid


def main():
    swarm = ["green", "red"]
    swarm_position = [None] * len(swarm)
    swarm_position_lpfs = []
    [swarm_position_lpfs.append(LowPassFilter(0.1)) for p in swarm_position]

    cap = cv2.VideoCapture(2)

    while True:
        _, frame = cap.read()

        ret, field = find_field_frame(frame)

        # Scale field to 720p
        if ret:
            field = cv2.resize(field, (1080, 720), interpolation=cv2.INTER_AREA)

            # Create empty occupancy grid
            occupancy_grid = np.zeros(shape=(field.shape[0], field.shape[1]), dtype=np.uint8)

            new_swarm_position = find_swarm_position(field, swarm=swarm)
            for i, bot_position in enumerate(new_swarm_position):
                if bot_position is not None:
                    swarm_position[i] = swarm_position_lpfs[i].update(bot_position).astype(int)

            print(swarm_position)

            # obstacles, obstacle_mask_viz = find_obstacles(field)
            # occupancy_grid = add_obstacles_to_occupancy_grid(occupancy_grid, obstacles)

            # return occupancy_grid

        if debug and ret:
            frame_resized = cv2.resize(frame, (1080, 720), interpolation=cv2.INTER_AREA)
            occupancy_grid_bgr = cv2.cvtColor(occupancy_grid, cv2.COLOR_GRAY2BGR)
            for point in swarm_position:
                if point is not None:    
                    cv2.circle(field, point[:2], 4, (0, 0, 255), -1)

            cv2.imshow("Original, Field", np.hstack([frame_resized, field]))
            # cv2.imshow("Occupancy Grid", occupancy_grid_bgr)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
