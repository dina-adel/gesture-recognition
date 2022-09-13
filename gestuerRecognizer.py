import cv2
import utils


# start the camera
video = cv2.VideoCapture(0)

# initialize variables
background = None
hand_image = None
defects = None

SHOW_CONTOURS = True  # show circle ROI and hand contour?
FINGER_NAMES = True  # show finger names?
FINGER_COUNT = False  # show finger count?

min_dist = 50  # used in detecting finger names
accum_weight_alpha = 0.5  # used in background averaging
top, right, bottom, left = 0, 0, 325, 280  # the part of the frame in which the hand is present
thresh = 25  # threshold used in detecting contours
count_over = 5  # make a prediction after how many frames?
cal_frames = 60  # number of frames to calculate background

step = -1

while True:

    step += 1

    if step % count_over == 0 and step > cal_frames:  # every 10 frames, apply the algorithm
        continue

    # read frame from camera
    ret, frame = video.read()
    frame = cv2.flip(frame, 1)

    # extract the hand part
    hand = frame[top:bottom, right:left]
    cv2.rectangle(frame, (left, top), (right, bottom), (255, 0, 0), 2)

    # transform to gray-scale and blur frame
    frame_gray = cv2.cvtColor(hand, cv2.COLOR_BGR2GRAY)
    frame_gray = cv2.GaussianBlur(frame_gray, (9, 9), 0)

    if background is None or step < cal_frames:
        background = utils.average_background(background, frame_gray, accum_weight_alpha)
        print(f'calibrating {step} ....')

    else:
        contours, hierarchy, threshold_frame = utils.get_contours(background, frame_gray, thresh)
        # find the biggest contour
        if len(contours) != 0:
            hand_contour = max(contours, key=cv2.contourArea)

            if SHOW_CONTOURS:  # show image contours
                cv2.imshow('hand', threshold_frame)
                cv2.drawContours(frame, [hand_contour + (right, top)], -1, (0, 255, 255))

            # Get Convex Hull & Defects
            try:
                # convex hull
                hand_convex_hull = cv2.convexHull(hand_contour)
                # get extreme points in the hull
                max_top, max_right, max_left, max_bottom = utils.get_max_hand_points(hand_convex_hull)
                # get the center of the detected hand contour
                center_x, center_y = utils.get_hand_center(max_top, max_right, max_left, max_bottom)
                # find defects in the hull
                defects = cv2.convexityDefects(hand_contour, cv2.convexHull(hand_contour, returnPoints=False,
                                                                            clockwise=True))
            except:
                # sometimes, especially when background is cluttered, defects can't be found
                print('No Defects')
                continue

            # Get the finger names
            if FINGER_NAMES:
                new_hull = cv2.convexHull(hand_contour, returnPoints=False, clockwise=True)
                new_defects = cv2.convexityDefects(hand_contour, new_hull)
                if new_defects is not None:
                    frame = utils.get_finger_name(hand_contour, new_defects, frame, center_x, center_y, min_dist)

            # Get the fingers count
            if FINGER_COUNT:
                frame = utils.count_fingers(frame, threshold_frame,
                                            center_x, center_y,
                                            max_left, max_right, max_top, max_bottom,
                                            SHOW_CONTOURS)

    # Display the resulting frame
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
video.release()
cv2.destroyAllWindows()
