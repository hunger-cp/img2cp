import itertools
import os
import pickle
import cv2
import matplotlib.pyplot as plt
import numpy as np
import math
from tqdm import tqdm
# from kmeans import *
# import houghbundler
from matplotlib import pyplot as plt

from skimage import data
from skimage.feature import corner_harris, corner_subpix, corner_peaks
from skimage.transform import warp, AffineTransform
from skimage.draw import ellipse


def identifyPoints(
        fileName: str,
        pointQuality: float,
        minDistance: int,
        top_crop: int = 0):
    """
    Identify possible point canddiates on a crease pattern image
    :param top_crop: top of image to crop off (for reference images, 55)
    :param fileName: name of image file
    :param pointQuality: point quality threshold, 0-1 by steps of .1
    :param minDistance: minimum distance between point candidates in pixels
    :return: (n, 2) list of n point candidate coordinates
    """
    # Import CP
    """IMAGE_PATH = "images/cp.png"  #@param {type:"string"}
    img = cv2.imread(IMAGE_PATH)"""
    img = cv2.imread('uploads/' + fileName)
    # img = cv2.resize(img, dsize=[4*i for i in img.shape[:-1]], fx=0, fy=0, interpolation=cv2.INTER_CUBIC)
    # Isolating Points if you need them
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)

    # Binary image (Needed for Hough Lines)
    # adapt_type = cv2.ADAPTIVE_THRESH_GAUSSIAN_C
    # thresh_type = cv2.THRESH_BINARY_INV
    # bin_img = cv2.adaptiveThreshold(gray, 255, adapt_type, thresh_type, 11, 2)
    _, bin_img = cv2.threshold(gray, 254, 255, cv2.THRESH_BINARY_INV)

    # Cropping image to square
    box = np.where(gray == 0)
    print(box)
    # crop = gray[box[0][0] - 1:box[0][-1] + 1, box[1][0] - 1:box[1][-1] + 1]
    # imgcrop = img[box[0][0] - 1:box[0][-1] + 1, box[1][0] - 1:box[1][-1] + 1]
    crop = gray[top_crop:]
    imgcrop = img[top_crop:]
    # Isolating lines if you need them
    # Find the edges in the image using canny detector
    rho, theta, t = 1, np.pi / 8, 500  # CPs dont have a lot of noise so we can use a low rho and threshold
    """
    lines = cv2.HoughLinesP(crop, rho=rho, theta=theta, threshold=t, minLineLength = 20, maxLineGap = 10)
    print(len(lines))
    """
    # print("bundling")
    # bundler = houghbundler.HoughBundler(min_distance=5, min_angle=11.25)
    # lines = bundler.process_lines(lines)
    """
    coords = point_peaks(point_harris(crop), min_distance=5, threshold_rel=0.05)
    coords_subpix = point_subpix(crop, coords, window_size=1)

    fig, ax = plt.subplots()
    ax.imshow(imgcrop, cmap=plt.cm.gray)
    ax.plot(coords[:, 1], coords[:, 0], color='cyan', marker='o',
            linestyle='None', markersize=6)
    ax.plot(coords_subpix[:, 1], coords_subpix[:, 0], '+r', markersize=15)
    ax.axis()
    plt.show()
    # Displaying lines
    """
    """
    dst = cv2.cornerHarris(np.float32(crop),2,1,0.04)
    dst = cv2.dilate(dst,None)
    ret, dst = cv2.threshold(dst,0.01*dst.max(),255,0)
    dst = np.uint8(dst)
    # find centroids
    ret, labels, stats, centroids = cv2.connectedComponentsWithStats(dst)
    # define the criteria to stop and refine the corners
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)
    points = cv2.cornerSubPix(crop,np.float32(centroids),(5,5),(-1,-1),criteria)
    for point in points:
        x,y = point.ravel()
        cv2.circle(imgcrop,(int(x),int(y)),5,(36,255,12),-1)
    plt.imshow(imgcrop)
    plt.show()
    """
    """
        for x in range(0, len(lines)):
            for x1,y1,x2,y2 in lines[x]:
                cv2.line(crop,(x1,y1),(x2,y2),(0),2)
        """
    """print(len(lines))
    segmented = segment_by_angle_kmeans(lines)
    print(len(segmented[1]))

    print(len(segmented[0]))
    for x in range(0, len(segmented[0])):
        for x1,y1,x2,y2 in lines[x]:
            cv2.line(imgcrop, (x1,y1),(x2,y2),(0,255,0),2)
    for x in range(0, len(segmented[1])):
        for x1,y1,x2,y2 in lines[x]:
            cv2.line(imgcrop, (x1,y1),(x2,y2),(0,0,0),2)
    """
    # For HoughLines Diplay
    """
    a,b,c = lines.shape
    for i in range(a):
        rho = lines[i][0][0]
        theta = lines[i][0][1]
        a = math.cos(theta)
        b = math.sin(theta)
        x0, y0 = a*rho, b*rho
        pt1 = ( int(x0+1000*(-b)), int(y0+1000*(a)) )
        pt2 = ( int(x0-1000*(-b)), int(y0-1000*(a)) )
        cv2.line(imgcrop, pt1, pt2, (0, 0, 255), 2, cv2.LINE_AA)
    """
    """
    def get_contours(img, crop):
        contours, hierarchies = cv2.findContours(crop, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        for cnt in contours:
            if cv2.contourArea(cnt) > 10:
                peri = cv2.arcLength(cnt, True)
                approx = cv2.approxPolyDP(cnt, peri * 0.04, True)
                cv2.drawContours(img, approx, -1, (0, 0, 255), 8)
    get_contours(imgcrop, crop)
    """
    # points = cv2.goodFeaturesToTrack(crop, 0, 0.1, 5)
    points = np.squeeze(cv2.goodFeaturesToTrack(crop, 0, pointQuality, minDistance))

    # VISUALIZE POINTS ON IMAGE
    print(points.shape)

    # Identify Corner Points
    # aprox corner points (acp)
    acp = [[box[0][0], box[1][0]],
           [box[0][0], box[1][-1]],
           [box[0][-1], box[1][0]],
           [box[0][-1], box[1][-1]]]
    corner_points = np.squeeze(np.array([findClosestPoint(points, apx_p)["point"] for apx_p in acp]))

    temp = []
    for point in points:
        matches = False
        for corner_point in corner_points:
            if (point == corner_point).all():
                matches = True
        if not matches:
            temp.append(point)

    points = np.array(temp)
    lower_left = corner_points[np.argmin([x + y for x, y in corner_points])]
    upper_right = corner_points[np.argmax([x + y for x, y in corner_points])]
    lower_left = [lower_left[0], lower_left[0]]
    upper_right = [upper_right[0], upper_right[0]]
    corner_points = [[lower_left[0], upper_right[0]], lower_left, upper_right, [upper_right[0], lower_left[0]]]
    points = np.concatenate((points, corner_points), axis=0)


    # edge lines
    lines = [[corner_points[0], corner_points[1]],
             [corner_points[0], corner_points[2]],
             [corner_points[1], corner_points[3]],
             [corner_points[2], corner_points[3]]]

    for point in points:
        cv2.circle(imgcrop, tuple(map(int, point)), 3, (36, 255, 12), -1)
    for corner_point in corner_points:
        cv2.circle(imgcrop, tuple(map(int, corner_point)), 5, (255, 36, 12), -1)
    for line in lines:
        cv2.line(imgcrop, tuple(map(int, line[0])), tuple(map(int, line[1])), (255, 36, 12), 3)
    for point in points:
        cv2.circle(imgcrop, tuple(map(int, point)), 3, (36, 255, 12), -1)
    print(corner_points)
    # [top left, bottom left, top right, bottom right]

    # scaled_points = scale(points=points, multiplier=mult, anchor=corner_points[1])
    # need to ultiamtely transform top left anchor point to -200, 200 but for now anchor it at 0, 200 for testing purposes
    print(points.shape)
    # points = np.squeeze(points, axis=1)
    lower_left = corner_points[np.argmin([x + y for x, y in corner_points])]
    upper_right = corner_points[np.argmax([x + y for x, y in corner_points])]
    mult = 400 / (upper_right[1]-lower_left[1])
    print(lower_left)
    print(upper_right)
    print(points)
    scaled_kref = kamiyaRefTester(points, 100, lower_left=corner_points[1], upper_right=corner_points[2])
    print(scaled_kref)
    # kref = scaled_kref["closest"]["point"] / mult + corner_points[1]
    # krefa = scaled_kref["ref"] / mult + corner_points[1]
    # print(kref)
    #print(scaled_kref)
    # yellow
    cv2.circle(imgcrop, tuple(map(int, scaled_kref["closest"]["point"])), 3, (255, 255, 12), -1)
    # pink
    cv2.circle(imgcrop, tuple(map(int, scaled_kref["ref"])), 3, (255, 12, 255), -1)
    # print(f"og ref: {scaled_kref['og_ref']}")
    og_scale = lambda k: k
    # og_scale = lambda k: k / mult + corner_points[1]
    cv2.line(imgcrop, tuple(map(int, og_scale(scaled_kref['og_ref'][0][0]))),
             tuple(map(int, og_scale(scaled_kref['og_ref'][0][1]))),
             (225, 12, 225), 3)
    cv2.line(imgcrop, tuple(map(int, og_scale(scaled_kref['og_ref'][1][0]))),
             tuple(map(int, og_scale(scaled_kref['og_ref'][1][1]))),
             (225, 12, 225), 3)
    print(f"The reference is the intersection between: {scaled_kref['kamiya_ref'][0]}_r x {scaled_kref['kamiya_ref'][1]}_l")
    # scale(points=points, multiplier=mult, anchor=lower_left)
    # transform(points=points, scale=(-200, -200))
    # print(points)

    #save all cv generated coordinates + starting coordinate (both actual and offset)
    with open('points.pickle', 'wb') as handle:
        pickle.dump(
            # (all_cv_generated_coor, offset_start_coor, actual_start_coor)
            (points, scaled_kref["ref"], scaled_kref["closest"]["point"], corner_points)
            , handle)

    return imgcrop

    # points = np.squeeze(points, axis=1)
    # print(points.tolist())
    # return points.tolist()


def transform(points, scale):
    for p in points:
        p[0] += scale[0]
        p[1] += scale[1]
    return points


def scale(points, multiplier, anchor):
    points = transform(points, [-anchor[0], -anchor[1]])
    for p in points:
        p *= multiplier
    # points = transform(points, anchor)
    return points


def midpoint(p1, p2):
    return (p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2


def kamiyaRefTester(points, thresh, lower_left=(-200, -200), upper_right=(200, 200)):
    # upperleft, lowerleft, upperight, lowerright
    upper_left, lower_right = (lower_left[0], upper_right[1]), (upper_right[0], lower_left[1])
    fnames = os.listdir("./img")
    ratioList = {"l": [], "r": []}
    # Loop thru all ref images in ./img and convert them to float ratios; split by l/r
    for f in fnames:
        f = f.split(".")[0]
        ratio = float('.'.join(f.split("_")[:-1]))
        ratioList[f.split("_")[-1]].append(ratio)
    # Find all l/r combos
    print("Finding combos...")
    combos = np.array(
        [r for r in itertools.product(ratioList['l'], ratioList['r'])])
    print(f"Combo Shape: {combos.shape}")
    # combos.reshape((combos.shape[0] * 2, 2))
    cline_l = lambda r: (lower_left, (r * (upper_right[0] - upper_left[0]) + upper_left[0], upper_right[1]))
    print("Finding intersections...")
    # store intersections in a dict to store original refs for a ref finder :D
    mirror_x = midpoint(lower_left, lower_right)[0]
    intersections = [{"intersection": intersect(flipHorizontal(cline_l(c[0]), scale_x=mirror_x), cline_l(c[1])),
                      "og": [flipHorizontal(cline_l(c[0]), scale_x=mirror_x), cline_l(c[1])], "kamiya_ref": c} for c in combos]
    print(f"Identifying candidates from {len(intersections)} intersections...")
    # "og_ref" stores the original reference (2 lines that intersect)
    # "ref" stores the actual coordinate
    # "closest" stores the closest point from the point-set
    # "kamiya_ref" returns the original kamiya reference
    candidates = [{"kamiya_ref": i["kamiya_ref"],
                   "og_ref": i["og"],
                   "ref": i["intersection"],
                   "closest": findClosestPoint(points, i["intersection"])}
                  for i in intersections]
    candidate = {"ref": [0, 0], "closest": {"point": [0, 0], "error": float('inf')}}
    print(f"Selecting optimal candidate from {len(candidates)} candidates")
    # perhaps the midpoints will be a temporary blacklist
    """blacklisted = [lower_left, upper_left, lower_right, upper_right,
                   (midpoint(lower_left, upper_right)),
                   (midpoint(lower_left, upper_left)),
                   (midpoint(lower_right, upper_right)),
                   (midpoint(lower_left, lower_right)),
                   (midpoint(upper_left, upper_right))]"""
    blacklisted = [lower_left, upper_left, lower_right, upper_right]
    # blacklisted = [[0, 0], [-200, -200], [0, 200], [200, 0], [200, 200], [-200, 0], [0, -200]]
    for c in candidates:
        # print(c["point"])
        # print(blacklisted)
        if True in [(c["closest"]["point"] == c_).all() for c_ in blacklisted]:
            continue
        if c["closest"]["error"] < candidate["closest"]["error"]:
            candidate = c
    if candidate["closest"]["error"] > thresh:
        return
    return candidate


def findClosestPoint(points, point):
    min_out = {"error": float('inf')}
    for p in points:
        error = dist(np.squeeze(p), point)
        if error < min_out["error"]:
            min_out = {"point": p, "error": error}
    return min_out


def findClosestPoints(points, point, thresh):
    out = {}
    for p in points:
        error = dist(p, point)
        if error < thresh:
            out[p] = error
    return out


def dist(p1, p2):
    return math.sqrt(math.pow(p1[0] - p2[0], 2) + math.pow(p1[1] - p2[1], 2))


def intersect(l1, l2):
    x1, y1, x2, y2, x3, y3, x4, y4 = l1[0][0], l1[0][1], l1[1][0], l1[1][1], l2[0][0], l2[0][1], l2[1][0], l2[1][1]
    m1 = (y2 - y1) / (x2 - x1)
    m2 = (y4 - y3) / (x4 - x3)
    x = (m1 * x1 - m2 * x3 - (y1 - y3)) / (m1 - m2)
    y = m1 * (x - x1) + y1
    return [x, y]


def flipHorizontal(l, scale_x=0):
    l = np.array(l)
    transform(l, (-scale_x, 0))
    l[0][0] = -l[0][0]
    l[1][0] = -l[1][0]
    transform(l, (scale_x, 0))
    return l
