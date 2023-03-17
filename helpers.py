import cv2
import pickle
import numpy as np
import math
import matplotlib.pyplot as plt
import backend_alg as bfunc
def tuple_float_to_int(t):
    replacement_tuple = []
    for i in range(len(t)):
        replacement_tuple.append(tuple(map(int, t[i])))
    return replacement_tuple
def upload_file():
    with open('points.pickle', 'rb') as handle:
        data = pickle.load(handle)
        points = tuple(map(tuple, data[0]))
        actual_ref = tuple(map(float, data[1]))
        offset_ref = tuple(map(int, data[2]))
        points = tuple_float_to_int(points)

        corner_points = data[3]
        corner_points = tuple(map(tuple, corner_points))
        corner_points = tuple_float_to_int(corner_points)
    return [points, offset_ref, actual_ref, corner_points]
def get_square_edges(corner_points):
    lower_left = corner_points[np.argmin([x + y for x, y in corner_points])]
    upper_right = corner_points[np.argmax([x + y for x, y in corner_points])]
    return (
        ((lower_left[0], lower_left[1]), (lower_left[0], upper_right[1])),
        ((lower_left[0], upper_right[1]), (upper_right[0], upper_right[1])),
        ((upper_right[0], upper_right[1]), (upper_right[0], lower_left[1])),
        ((upper_right[0], lower_left[1]), (lower_left[0], lower_left[1])))
def add_pixels(c1, c2):
    return c1[0] + c2[0], c1[1] + c2[1], c1[2] + c2[2]
def clarify_BGR(image):
    for x in range(image.shape[1]):
        for y in range(image.shape[0]):
            color = image[y][x]
            if color[2] > color[0] + 100 and color[2] > color[1] + 100:
                image[y][x] = (0, 0, 255)
            elif color[0] > color[1] + 100 and color[0] > color[2] + 100:
                image[y][x] = (255, 0, 0)
            elif color[0] < 60 and color[1] < 60 and color[2] < 60:
                image[y][x] = (0, 0, 0)
            else:
                image[y][x] = (255, 255, 255)
def py2round(data):
    if(data == 0):
        return 0
    if(abs(data) - int(abs(data)) > .5):
        if data > 0:
            return int(math.ceil(abs(data)))
        else:
            return int(math.ceil(abs(data)) * -1)
    else:
        if data < 0:
            return int(math.floor(abs(data)) * -1)
        else:
            return int(math.floor(abs(data)))
def init_images(im_name, points):
    im = cv2.imread(im_name, cv2.IMREAD_COLOR)
    height, width = im.shape[0], im.shape[1]
    clarify_BGR(im)
    debug_im = np.ones((height, width, 3)) * 255
    for point in points:
        cv2.circle(im, point, 3, (0, 0, 0), 1)
    return im, debug_im
def create_plot(images):
    fig = plt.figure(figsize=(10,7))
    for i in range(len(images)):
        swap_rgb_bgr(images[i])
        fig.add_subplot(1, len(images), i + 1)
        plt.imshow(images[i])
    return fig
def swap_rgb_bgr(im):
    height, width = im.shape[0], im.shape[1]
    for x in range(width):
        for y in range(height):
            im[y][x] = (im[y][x][2], im[y][x][1], im[y][x][0])
def combine_images(over, under):
    combined = over.copy()
    height, width = over.shape[0], over.shape[1]
    for x in range(width):
        for y in range(height):
            if (over[y][x] == (255, 255, 255)).all():
                combined[y][x] = under[y][x]
    return combined
def check_found_links(im_name, points, offset_ref, actual_ref, corner_points): #(im_name, points, corner_points)
    images = init_images(im_name, points)
    bfunc.get_links(points, images[0], images[1])
    bfunc.check_links(im_name, points)

    # check_found_links_v2 (can check for specific points)
    # for point in points:
    #     images = init_images(im_name, points)
    #     im, debug_im = images[0], images[1]
    #     for up in points:
    #         is_line(point, up, im, debug_im)
    #     im_combined = combine_images(debug_im, im)
    #     fig = create_plot([im, im_combined, debug_im])
    #     plt.show()
def gl(im_name, points, offset_ref, actual_ref, corner_points):
    images = init_images(im_name, points)
    bfunc.get_links(points, images[0], images[1])
    bfunc.get_lines(points, actual_ref, offset_ref, corner_points, images[0], images[1], im_name)


CHECK_COLOR_WIDTH = 4
CHECK_COLOR_HEIGHT = 2
CHECK_COLOR_DIV = 25
PERCENT_MATCHING_COLOR = 1/10000
ANGLE_THRESH = .3 # 0-.5 right now but needs to vary with distance -> use kx^2 relationship

# 0 = black
# 1 = red
# 2 = blue
# 3 = white
def get_line_color (start, end, image):
    dx, dy = (end[0] - start[0]) + 0.0000000000001, (end[1] - start[1]) - 0.0000000000001
    c = math.sqrt(1/(dx**2 + dy**2)) # multiply by constant factor to get scalar magnitude of 1 for |<dx, dy>|
    perp_dx, perp_dy = (start[1] - end[1])*c, (end[0] - start[0])*c
    div = CHECK_COLOR_DIV # number of intervals to consider between points
    dx/=div
    dy/=div

    sum_BGR = [0,0,0]
    i, x, y = 0, start[0] + 2*dx, start[1] + 2*dy
    while i < div - 2:
        # set the width/height of region where pixels are being considered
        width = CHECK_COLOR_WIDTH
        height = CHECK_COLOR_HEIGHT

        white = True
        for j in range(-1*width, width+1):
            for k in range(-height, height+1):
                sum_BGR = add_pixels(sum_BGR, image[int(y + perp_dy*j + dy*c*k*div)][int(x + perp_dx*j + dx*c*k*div)])
                if (image[int(y + perp_dy*j + dy*c*k*div)][int(x + perp_dx*j + dx*c*k*div)] != (255, 255, 255)).any():
                    white = False
        if white:
            return 3

        x += dx; y += dy; i += 1

    sum_BGR = (sum_BGR[0]/255, sum_BGR[1]/255, sum_BGR[2]/255)

    num_checked_pts = (div-2) * (width * 2 + 1) * (height * 2 + 1)
    min_matching_color = num_checked_pts * PERCENT_MATCHING_COLOR # value in parentheses is % of points that have to match given color
    if sum_BGR[2] > sum_BGR[0] + min_matching_color and sum_BGR[2] > sum_BGR[1] + min_matching_color:
        return 1
    if sum_BGR[0] > sum_BGR[1] + min_matching_color and sum_BGR[0] > sum_BGR[2] + min_matching_color:
        return 2
    if sum_BGR[0] < num_checked_pts - min_matching_color and sum_BGR[1] < num_checked_pts - min_matching_color and sum_BGR[2] < num_checked_pts - min_matching_color:
        return 0
    return 3
#returns 0 - 31
def is_line(start, end, image, debug_image):
    start = (start[0], start[1])
    end = (end[0], end[1])
    if start == end:
        return -100

    angle = get_angle(start, end)

    if abs(angle-py2round(angle)) <= ANGLE_THRESH:  # can change threshold
        # debug_image = cv2.line(debug_image, start, end, (0, 0, 0), 1)
        if get_line_color(start, end, image) != 3:
            set_line_color(start, end, image, debug_image)
            return angle
    return -100
def get_angle(start, end):
    dx = end[0] - start[0] + 0.000001
    dy = end[1] - start[1]
    angle = math.degrees(math.atan2(dy,dx))
    angle /= 11.25
    return (py2round(angle) + 32) % 32
def set_line_color(start, end, image, debug_image):
    glc = get_line_color(start, end, image)
    if glc == 0:  # black
        debug_image = cv2.line(debug_image, start, end, (0, 0, 0), 5)
    elif glc == 1:  # red
        pass
        debug_image = cv2.line(debug_image, start, end, (0, 0, 255), 5)
    elif glc == 2:  # blue
        debug_image = cv2.line(debug_image, start, end, (255, 0, 0), 5)
    elif glc == 3:  # white
        debug_image = cv2.line(debug_image, start, end, (0, 255, 0), 3)