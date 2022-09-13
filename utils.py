import math
import cv2
import numpy as np
from sklearn.metrics import pairwise


def dist(a, b):
    """ Calculates the euclidean distance between two points
        Inputs:
            - a (tuple): point a (x,y)
            - b (tuple): point x (x,y)
        Returns: the distance between the two points
    """
    return math.sqrt((a[0] - b[0]) ** 2 + (b[1] - a[1]) ** 2)


def average_background(background, frame_gray, accum_weight_alpha):
    """ Used to calculate the Average weighted background
    Inputs:
            - background (frame): the previous background
            - frame_gray (frame): the current frame
            - accum_weight_alpha (float): accumulation alpha for averaging
        Returns: the updated background
    """
    if background is None:
        background = frame_gray.copy().astype("float")
    else:
        background = cv2.accumulateWeighted(frame_gray, background, accum_weight_alpha)
    return background


def get_contours(background, frame_gray, thresh):
    """ Gets the contours from the upcoming frames
    Inputs:
        - background (frame): the average background obtained before
        - frame_gray (frame): the new frame
        - thresh: threshold for frame to get contours

        returns: (tuple) contours, hierarchy, threshold_frame
    """

    # the new frame is subtracted from the background
    normalize_background = background.copy().astype("uint8")
    foreground = cv2.absdiff(normalize_background, frame_gray)

    # making the image/frame with a binary threshold (black and white) for the contours
    threshold_frame = cv2.threshold(foreground, thresh, 255, cv2.THRESH_BINARY)[1]
    contours, hierarchy = cv2.findContours(threshold_frame, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    return contours, hierarchy, threshold_frame


def get_hand_center(max_top, max_right, max_left, max_bottom):
    """ Get the expected center of the hand based on the contour extreme points
    """
    # find the center of the hand
    center_x = int((max_left[0] + max_right[0]) / 2)
    center_y = int((max_top[1] + max_bottom[1]) / 2)
    return center_x, center_y


def get_max_hand_points(hand_convex_hull):
    """ Get the boundaries of the hand (extreme left, right, top, and bottom)
    """
    # find the extreme points indices
    extreme_top = hand_convex_hull[:, :, 1].argmin()
    extreme_bottom = hand_convex_hull[:, :, 1].argmax()

    extreme_left = hand_convex_hull[:, :, 0].argmin()
    extreme_right = hand_convex_hull[:, :, 0].argmax()

    # the max
    max_top = tuple(hand_convex_hull[extreme_top][0])
    max_bottom = tuple(hand_convex_hull[extreme_bottom][0])
    max_left = tuple(hand_convex_hull[extreme_left][0])
    max_right = tuple(hand_convex_hull[extreme_right][0])

    return max_top, max_right, max_left, max_bottom


def get_finger_name(hand_contour, defects, frame, center_x, center_y, min_distance):
    """ Get the name of the finger based on the angle between the hand-axis and the points
        Inputs:
            - hand_contour: the hand contour obtained in previous steps
            - defects: defect points obtained
            - frame: the original frame we manipulate
            - center_x, center_y: center of the detected hand
            - min_distance: used in filtering defects' points; the minimum distance acceptable between two points
        returns: Writes the finger names on the video stream (frame)
    """
    points = get_points(hand_contour, defects)  # get defect points
    points = filter_points(points, min_distance)  # filter points based on distance
    for end in points:

        my_radians = math.atan2(center_x - end[0], center_y - end[1])
        my_degrees = math.degrees(my_radians)
        my_degrees = int(my_degrees)

        if my_degrees > 45 and my_degrees < 62:
            cv2.putText(frame, "Pinky", end, cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0),
                        2, cv2.LINE_AA)
            continue

        if my_degrees > 20 and my_degrees < 46:
            cv2.putText(frame, "Ring", end, cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 0),
                        2, cv2.LINE_AA)
            continue

        if my_degrees < -76:
            cv2.putText(frame, "Thumb", end, cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255),
                        2, cv2.LINE_AA)
            continue

        if my_degrees > -37 and my_degrees < -25:
            cv2.putText(frame, "Index", end, cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 0),
                        2, cv2.LINE_AA)
            continue

        if my_degrees > -3 and my_degrees < 14:
            cv2.putText(frame, "Middle", end, cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 255),
                        2, cv2.LINE_AA)
            continue
    return frame


def get_points(hand_contour, defects):
    """ Gets the points that define the defects in the hand contour.
        We mainly care about the points that can represent a fingertip
        Inputs:
            - hand_contour: the hand contour
            - defects: the defects obtained from the convex hull and the hand contour
        returns (list): points : represent the expected fingertips
    """
    points = []
    for i in range(defects.shape[0]):
        # s: start, e:end, f:farthest point, d:distance to f
        s, e, f, d = defects[i, 0]
        end = tuple(hand_contour[e][0])
        points.append(end)
    return points


def filter_points(points, filterValue):
    """ Filter irrelevant points based on the distance between them
        Inputs:
            - points (list): points obtained from the defects
            - filterValue (int): the minimum distance between two points
        returns: the filtered points
    """

    filtered = [points[0]]  # initialize with the first point

    for point_i in filtered:
        if len(filtered) > 10:
            break
        for point_j in points:
            if point_i == point_j:
                continue
            # here
            if dist(point_i, point_j) > filterValue:
                filtered.append(point_j)

    return filtered


def count_fingers(frame, threshold_frame,
                  center_x, center_y,
                  max_left, max_right, max_top, max_bottom,
                  SHOW_CONTOURS=False):

    """ Count the number fingers raised in the frame
        Inputs:
            - frame: the original frame we show at the end
            - threshold_frame: the black & white hand frame
            - max_left, max_right, max_top, max_bottom: the boundaries of the contour
            - SHOW_CONTOURS: whether to show the window with the circle ROI or the contour window

    """
    # Assumption: the longest distance represents the longest finger
    distance = pairwise.euclidean_distances([(center_x, center_y)],
                                            Y=[max_left, max_right, max_top, max_bottom])[0]
    max_distance = distance[distance.argmax()]

    # calculate the mask circle
    mask_circle_radius = int(0.7 * max_distance)
    circumference = (2 * np.pi * mask_circle_radius)

    # draw the circle
    mask_circle = np.zeros(threshold_frame.shape[:2], dtype="uint8")
    cv2.circle(mask_circle, (center_x, center_y), mask_circle_radius, 255, 1)

    # Anding the circle with the image to mask out the fingers
    mask_circle = cv2.bitwise_and(threshold_frame, threshold_frame, mask=mask_circle)

    if SHOW_CONTOURS:
        cv2.imshow('circular_roi', mask_circle)

    (cnts, _) = cv2.findContours(mask_circle.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    count = 0
    # the number of fingers is related to the number of times a finger intersects with the circle
    for c in cnts:
        (x, y, w, h) = cv2.boundingRect(c)
        if ((center_y + (center_y * 0.25)) > (y + h)) and ((circumference * 0.25) > c.shape[0]):
            count += 1

    if count > 5: # bound the number of fingers
        count = 5

    cv2.putText(frame, '# of fingers: ' + str(count), (10, 450), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 2,
                cv2.LINE_AA)

    return frame
