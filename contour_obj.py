import math
import polygon_triangulate

def obj_from_contour(contour_points):
    si = -1
    ei = len(contour_points) - 1
    mid_point_index = math.ceil(len(contour_points)/2) - 1
    interleaved_points = []
    while si != mid_point_index:
        si += 1
        interleaved_points.append(contour_points[si])
        if ei > mid_point_index:
            interleaved_points.append(contour_points[ei])
        ei -= 1
    obj_str = ""
    for p in interleaved_points:
        obj_str += "v {0} {1} 0\n".format(p[0], p[1])
    for p in interleaved_points:
        obj_str += "v {0} {1} -1\n".format(p[0], p[1])
    reverse = False
    for pi in range(2, len(interleaved_points)):
    # for pi in range(2, 3):
        face = []
        face.append(pi - 1)
        face.append(pi)
        if reverse:
            face.reverse()
        face.append(pi+1)
        reverse = not reverse
        obj_str += "f {0} {1} {2}\n".format(face[0], face[1], face[2])
        face.reverse()
        obj_str += "f {0} {1} {2}\n".format(face[0] + len(interleaved_points), face[1] + len(interleaved_points), face[2] + len(interleaved_points))

    for pi in range(1, len(contour_points)):
    # for pi in range(1, 2):
        first_point = interleaved_points.index(contour_points[pi - 1]) + 1
        second_point = interleaved_points.index(contour_points[pi]) + 1
        third_point = second_point + len(interleaved_points)
        fourth_point = first_point + len(interleaved_points)
        obj_str += "f {0} {1} {2}\n".format(first_point, second_point, third_point)
        obj_str += "f {0} {1} {2}\n".format(first_point, third_point, fourth_point)
    first_point = 2
    second_point = 1
    third_point = second_point + len(interleaved_points)
    fourth_point = first_point + len(interleaved_points)
    obj_str += "f {0} {1} {2}\n".format(first_point, second_point, third_point)
    obj_str += "f {0} {1} {2}\n".format(first_point, third_point, fourth_point)
    return(obj_str)

def obj_triang(contour_points, re_x, re_y):
    obj_str = ""
    faces = polygon_triangulate.polygon_triangulate(len(contour_points), [x[0] for x in contour_points], [y[1] for y in contour_points])
    for p in contour_points:
        obj_str += "v {0} {1} 0\n".format(p[0]*0.1, p[1]*0.1)
        obj_str += "vt {0} {1}\n".format(p[0]/re_x, 1 - p[1]/re_y*-1)
    for p in contour_points:
        obj_str += "v {0} {1} -5\n".format(p[0]*0.1, p[1]*0.1)
        obj_str += "vt {0} {1}\n".format(p[0]/re_x, 1 - p[1]/re_y*-1)

    for f in faces:
        obj_str += "f {0}/{0} {1}/{1} {2}/{2}\n".format(f[0]+1, f[1]+1, f[2]+1)
        f = f.tolist()
        f.reverse()
        obj_str += "f {0}/{0} {1}/{1} {2}/{2}\n".format(f[0]+1+len(contour_points), f[1]+1+len(contour_points), f[2]+1+len(contour_points))
    
    for pi in range(1, len(contour_points)):
        first_point = pi
        obj_str += "vt {0} {1}\n".format(1/len(contour_points) * (pi - 1), 0)
        second_point = pi + 1
        obj_str += "vt {0} {1}\n".format(1/len(contour_points) * (pi), 0)
        third_point = second_point + len(contour_points)
        obj_str += "vt {0} {1}\n".format(1/len(contour_points) * (pi - 1), 0.2)
        fourth_point = first_point + len(contour_points)
        obj_str += "vt {0} {1}\n".format(1/len(contour_points) * (pi), 0.2)
        obj_str += "f {0}/{3} {1}/{4} {2}/{5}\n".format(third_point, second_point, first_point, len(contour_points)*2 + 3, len(contour_points)*2 + 2, len(contour_points)*2 +1)
        obj_str += "f {0}/{3} {1}/{4} {2}/{5}\n".format(fourth_point, third_point, first_point, len(contour_points)*2 + 4, len(contour_points)*2 + 3, len(contour_points)*2 +1)
    first_point = len(contour_points)
    second_point = 1
    third_point = second_point + len(contour_points)
    fourth_point = first_point + len(contour_points)
    obj_str += "f {0}/{0} {1}/{1} {2}/{2}\n".format(third_point, second_point, first_point)
    obj_str += "f {0}/{0} {1}/{1} {2}/{2}\n".format(fourth_point, third_point, first_point)
    return obj_str

if __name__ == "__main__":
    contour_points = [[2, 1], [4, 1], [6, 0], [5, -1], [3, -1], [1, 0]]
    contour_points.reverse()
    obj_str = obj_triang(contour_points)
    with open('test.obj'.format(), 'w+') as wf:
        wf.write(obj_str)