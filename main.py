import helpers as func
import backend_alg as bfunc
import functions

im_name = r'./uploads/cp.png'
data = functions.identifyPoints(fileName=im_name, pointQuality=0.1, minDistance=5)
points, actual_ref, offset_ref, corner_points = data[0], data[1], data[2], data[3]
func.gl(im_name, points, offset_ref, actual_ref, corner_points)
# func.check_found_links(im_name, points, offset_ref, actual_ref, corner_points)

images = func.init_images(im_name, points)
bfunc.show_solution(images[0], images[1])

# original method for getting_lines (doesn't work but saving it here)
"""
def get_lines(points, offset_pt, kwn_pt, corners, image, debug_image, im_name):
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
        for kp in unsolved_kpts:
            if debug:
                print("################################################################################### KNOWN POINT:", kp, "###################################################################################")
                print("ACT2OFF:", act2off)
                print("UNSOLVED_KPTS:", len(unsolved_kpts), unsolved_kpts)
                images = func.init_images(im_name, points)
                image, debug_image = images[0], images[1]
                for line in lines:
                    debug_image = cv2.line(debug_image, tuple(int(num) for num in line[0][0]), tuple(int(num) for num in line[0][1]), (0, 0, 0), 5)
            links_to_remove = []
            if debug:
                print("\nLINKS_TO_REMOVE: (known point) (unknown endpoint) ((line_start1) (line_start2)) (calculated_intersection)")
            lines_to_add = []
            print(kp, tuple(act2off[tuple(kp)]))

            for link in links[tuple(act2off[tuple(kp)])]:
                for line in lines:
                    intersection = intersect(kp, link, line)
                    if not intersection == None:
                        solvable = True
                        # print("%s, %s, %s, %s" % (kp, link, line[0], line[1]))

                        # don't remove more than once if multiple lines intersect at the same endpoint
                        if not link in links_to_remove:
                            links_to_remove.append(link)

                        # updating act2off
                        already_found = False
                        for key in act2off.keys():
                            if math.dist(key, intersection) <= INTERSECT_THRESH:
                                already_found = True
                                intersection = key
                                break
                        if not already_found:
                            act2off[intersection] = link[0]

                        #see if line already exists
                        line_exists = False
                        for l in lines_to_add:
                            if l[0] == (kp, intersection):
                                line_exists = True
                        if not line_exists:
                            lines_to_add.append(((kp, intersection), func.py2round(func.get_angle(kp, intersection))))
                            if debug:
                                print(kp, link[0], line[0], intersection)

            if debug:
                print("\nREMOVING")
                print("***BEFORE:", kp, links[tuple(act2off[tuple(kp)])])
            for link in links_to_remove: # duplicate variable name "link" from outer loop
                links[tuple(act2off[tuple(kp)])].remove(link)

                rem_links = []
                for l in links[link[0]]:
                    if l[0] != tuple(act2off[tuple(kp)]):
                        rem_links.append(l)
                links[link[0]] = rem_links
                if debug:
                    print("REMOVED START FROM ENDPOINT", link[0], links[link[0]])
                #done removing solved links
########################################################################################################################################
                # remove points whose links have all been solved for
                if len(links[link[0]]) == 0:
                    del links[link[0]]
                    act_pt = (-1,-1)
                    for key in act2off.keys():
                        if act2off[key] == link[0]:
                            act_pt = key
                            kp_to_remove.append(key)
                            print("KEY REMOVED:", key)
                    # while tuple(act2off[tuple(key)]) in unsolved_kpts:
                    #     unsolved_kpts.remove(key)
########################################################################################################################################
            if len(links[tuple(act2off[tuple(kp)])]) == 0:
                del links[tuple(act2off[tuple(kp)])]
            # done removing solved points

            if debug:
                print("\nLINES TO ADD:")
            for line in lines_to_add:
                if debug:
                    debug_image = cv2.line(debug_image, tuple(int(num) for num in line[0][0]), tuple(int(num) for num in line[0][1]), (0, 0, 255), 5)
                    print(line)
                # check if the line has been found via other endpoint already
                if not (line[0][1], line[0][0]) in lines:
                    lines.add(line)
                    print(line[0][1])
                # add intersections with unsolved links to stack
                if tuple(act2off[tuple(line[0][1])]) in links:
                    # if len(tuple(act2off[tuple(line[0][1])])) > 0:
                        if not tuple(line[0][1]) in unsolved_kpts: # might need to check for approximate equals
                            unsolved_kpts.append(tuple(line[0][1]))
########################################################################################################################################
            if not tuple(act2off[tuple(kp)]) in links: # if kp has been removed because it has been fully solved
                # while tuple(act2off[tuple(kp)]) in unsolved_kpts:
                #     unsolved_kpts.remove(kp)
                kp_to_remove.append(kp)
########################################################################################################################################
            if debug:
                if tuple(act2off[tuple(kp)]) in links:
                    print("***AFTER:", kp, links[tuple(act2off[tuple(kp)])])
                else:
                    print("***AFTER:", kp, "solved")
                print("\nKNOWN LINES:")
                for line in lines:
                    print(line)
                print("\nREMAINING LINKS: (point) [links]")
                for key in links.keys():
                    print(key, links[key])
                im_combined = func.combine_images(debug_image, image)
                fig = func.create_plot([image, im_combined, debug_image])
                plt.show()

        for pt in kp_to_remove:
            print("REMOVING SOLVED POINTS", pt, act2off[tuple(pt)], unsolved_kpts)
            while tuple(act2off[tuple(pt)]) in unsolved_kpts:
                unsolved_kpts.remove(pt)
                if debug:
                    print("REMOVING SOLVED POINTS", pt, unsolved_kpts)
        if not solvable:
            print("\nunsolvable")
            break
"""
