import cv2
import math
import helpers as func
import matplotlib.pyplot as plt

ENDPOINT_ON_LINE_SEGMENT_THRESH = .95 # 0-1 (greater than thresh is valid line)
INTERSECT_THRESH = 2 # pixels away from a known point to be considered the same point
# CATCH_PARALLEL_THRESH = 1 # currently unused, reintroduce if angle comparison doesn't work

links = {} # between offset_points
act2off = {} # {actual: offset}
lines = [] # (start, end)
unsolved_kpts = []
def get_links(points, image, debug_image):
    for start in points:
        for end in points:
            angle = func.is_line(start, end, image, debug_image)
            if not angle == -100:
                if tuple(start) in links:
                    links[tuple(start)].append((end, angle))
                else:
                    links[tuple(start)] = [(end, angle)]

debug = False
def get_lines(points, offset_pt, kwn_pt, corners, image, debug_image, im_name):
    global unsolved_kpts, lines, links, act2off

    unsolved_kpts.append(kwn_pt)
    act2off[tuple(kwn_pt)] = offset_pt
    lines = set()
    for line in func.get_square_edges(corners):
        lines.add((line, func.py2round(func.get_angle(line[0], line[1]))))
    for point in corners:
        act2off[point] = point

    while len(links) > 0:
        solvable = False
        kp_to_remove = []
        links_to_remove = []
        lines_to_add = []
        for kp in unsolved_kpts:
            if debug:
                print("################################################################################### KNOWN POINT:", kp, "###################################################################################")
                print("ACT2OFF:", act2off)
                print("UNSOLVED_KPTS:", len(unsolved_kpts), unsolved_kpts)
                images = func.init_images(im_name, points)
                image, debug_image = images[0], images[1]
                for line in lines:
                    debug_image = cv2.line(debug_image, tuple(int(num) for num in line[0][0]), tuple(int(num) for num in line[0][1]), (0, 0, 0), 5)

            if debug:
                print("\nLINKS_TO_REMOVE: (known point) (unknown endpoint) ((line_start1) (line_start2)) (calculated_intersection)")
            for link in links[tuple(act2off[tuple(kp)])]:
                for line in lines:
                    intersection = intersect(kp, link, line)
                    if not intersection == None:
                        solvable = True

                        # don't remove more than once if multiple lines intersect at the same endpoint
                        if not (link, kp) in links_to_remove:
                            links_to_remove.append((link, kp))

                        # updating act2off
                        already_found = False
                        for key in act2off.keys():
                            if math.dist(key, intersection) <= INTERSECT_THRESH:
                                already_found = True
                                intersection = key
                                break
                        if not already_found:
                            act2off[intersection] = link[0]

                        # see if line already exists
                        line_exists = False
                        for l in lines:
                            if l[0] == (kp, intersection):
                                line_exists = True
                        for l in lines_to_add:
                            if l[0] == (kp, intersection):
                                line_exists = True
                        if not line_exists:
                            lines_to_add.append(((kp, intersection), func.py2round(func.get_angle(kp, intersection))))
                            if debug:
                                print(kp, link[0], line[0], intersection)

        if debug:
            print("\nREMOVING")
            # print("***BEFORE:", kp, links[tuple(act2off[tuple(kp)])])
        for link in links_to_remove: # (link, kp)
            if debug:
                print("HERE", link, tuple(act2off[tuple(link[1])]))
            if tuple(act2off[tuple(link[1])]) in links:
                if link[0] in links[tuple(act2off[tuple(link[1])])]:
                    links[tuple(act2off[tuple(link[1])])].remove(link[0])

            if link[0][0] in links:
                rem_links = []
                for l in links[link[0][0]]:
                    if l[0] != tuple(act2off[tuple(link[1])]):
                        rem_links.append(l)
                links[link[0][0]] = rem_links
                if debug:
                    print("REMOVED START FROM ENDPOINT", link[0][0], links[link[0][0]])
            # done removing solved links

                # remove points whose links have all been solved for
                # if the point hasn't already been removed
                if len(links[link[0][0]]) == 0:
                    del links[link[0][0]]
                    for key in act2off.keys():
                        if act2off[key] == link[0][0]:
                            kp_to_remove.append(key)
                            if debug:
                                print("KEY TO REMOVE:", key)

            for kp in unsolved_kpts:
                if tuple(act2off[tuple(kp)]) in links:
                    if len(links[tuple(act2off[tuple(kp)])]) == 0:
                        del links[tuple(act2off[tuple(kp)])]
                        kp_to_remove.append(kp)
            # done removing solved points

        if debug:
            print("\nLINES TO ADD:")
        for line in lines_to_add:
            if debug:
                debug_image = cv2.line(debug_image, tuple(int(num) for num in line[0][0]),
                                       tuple(int(num) for num in line[0][1]), (0, 0, 255), 5)
                print(line)

            # check if the line has been found via other endpoint already
            if not (line[0][1], line[0][0]) in lines:
                    lines.add(line)

            # add intersections with unsolved links to stack
            if tuple(act2off[tuple(line[0][1])]) in links:
                # if len(tuple(act2off[tuple(line[0][1])])) > 0:
                if not tuple(line[0][1]) in unsolved_kpts:  # might need to check for approximate equals
                    unsolved_kpts.append(tuple(line[0][1]))

        if debug:
            print("KP_TO_REMOVE:", kp_to_remove)
        for pt in kp_to_remove:
            # while tuple(act2off[tuple(pt)]) in unsolved_kpts:
            unsolved_kpts = [x for x in unsolved_kpts if x != pt]
            if debug:
                print("REMOVING SOLVED POINTS", pt, act2off[tuple(pt)], unsolved_kpts)

        if debug:
            print("\nKNOWN LINES:")
            for line in lines:
                print(line)
            print("\nREMAINING LINKS: (point) [links]")
            for key in links.keys():
                print(key, links[key])
            im_combined = func.combine_images(debug_image, image)
            fig = func.create_plot([image, im_combined, debug_image])
            plt.show()

        if not solvable:
            print("\nunsolvable")
            break

    """
    AT LINE 89 if debug: "REMOVING" in main.py
    """
# start1 = known point, end1 = (unknown point, angle), start/end 2 = known line
# angle is from 0-15 measuring 11.25 intervals from 0-180 degrees
def intersect(start1, link, line):
    end1, link_angle = link[0], link[1]
    line_segment, ls_angle = line[0], line[1]
    start2, end2 = line_segment[0], line_segment[1]

    if not math.dist(start2, end2) / (math.dist(end1, start2) + math.dist(end1, end2)) >= ENDPOINT_ON_LINE_SEGMENT_THRESH:
        return None
    if ls_angle == link_angle or abs(ls_angle-link_angle) == 16:
        return None

    link_angle = math.radians(link_angle*11.25)
    # calculate angle and create new x2 y2 points using dy and dx of 11.25 angles to get exact intersection
    end1 = (start1[0] + math.cos(link_angle), start1[1] + math.sin(link_angle))

    x1, y1, x2, y2, x3, y3, x4, y4 = start1[0], start1[1], end1[0], end1[1], start2[0], start2[1], end2[0], end2[1]

    # if abs((x1-x2)*(y3-y4)-(y1-y2)*(x3-x4)) <= CATCH_PARALLEL_THRESH:
    #     return None

    x = ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)) / (
                (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4))
    y = ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)) / (
                (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4))
    return (x, y)

def check_links(im_name, points):
    print("\nCHECKING LINKS")
    for key in links.keys():
        print(key, links[key])
        key = int(key[0]), int(key[1])
        images = func.init_images(im_name, points)
        im, debug_im = images[0], images[1]
        for link in links[key]:
            link = link[0][0], link[0][1]
            debug_im = cv2.line(debug_im, key, link, (0, 0, 0), 1)
        im_combined = func.combine_images(debug_im, im)
        fig = func.create_plot([im, im_combined, debug_im])
        plt.show()

def show_solution(image, debug_image):
    global lines, act2off

    for line in lines:
        print(line[0][0], line[0][1])
        debug_image = cv2.line(debug_image, [int(x) for x in line[0][0]], [int(x) for x in line[0][1]], (0, 0, 255), 5)
        # func.set_line_color([int(x) for x in line[0][0]], [int(x) for x in line[0][1]], image, debug_image)
    print(len(lines))
    fig = plt.figure(figsize=(10, 7))
    fig.add_subplot(1, 1, 1)
    func.swap_rgb_bgr(debug_image)
    plt.imshow(debug_image)
    plt.show()