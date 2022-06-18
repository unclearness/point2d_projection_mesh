import json
import time
import numpy as np
import trimesh
import projection as prj


def _make_ply_txt(vertices, faces, color=[], normal=[]):
    header_lines = ["ply", "format ascii 1.0",
                    "element vertex " + str(len(vertices)),
                    "property float x", "property float y", "property float z"]
    has_normal = len(vertices) == len(normal)
    has_color = len(vertices) == len(color)
    if has_normal:
        header_lines += ["property float nx",
                         "property float ny", "property float nz"]
    if has_color:
        header_lines += ["property uchar red", "property uchar green",
                         "property uchar blue", "property uchar alpha"]
    header_lines += ["element face " + str(len(faces)),
                     "property list uchar int vertex_indices", "end_header"]
    header = "\n".join(header_lines) + "\n"

    data_lines = []
    for i in range(len(vertices)):
        line = [vertices[i][0], vertices[i][1], vertices[i][2]]
        if has_normal:
            line += [normal[i][0], normal[i][1], normal[i][2]]
        if has_color:
            line += [int(color[i][0]), int(color[i][1]), int(color[i][2]), 255]
        line_txt = " ".join([str(x) for x in line])
        data_lines.append(line_txt)
    for f in faces:
        line_txt = " ".join(['3'] + [str(int(x)) for x in f])
        data_lines.append(line_txt)

    data_txt = "\n".join(data_lines)

    ply_txt = header + data_txt

    return ply_txt


def writeMeshAsPly(path, vertices, faces):
    with open(path, 'w') as f:
        txt = _make_ply_txt(vertices, faces)
        f.write(txt)


def main():
    # Load mesh
    # Decimated mesh
    # mesh_path = './data/max-planck_10k.obj'
    # Original mesh
    mesh_path = './data/max-planck.obj'
    mesh = trimesh.load(mesh_path)

    # Load camera param
    with open('./data/camera_param.json') as fp:
        camera_param = json.load(fp)
    K = camera_param['K']
    fx = K[0][0]
    fy = K[1][1]
    cx = K[0][2]
    cy = K[1][2]
    w2c_R = np.array(camera_param['R_world2cv'])
    w2c_t = np.array(camera_param['T_world2cv'])
    c2w_R = w2c_R.T
    c2w_t = -1 * c2w_R.dot(w2c_t)

    # Load landmarks
    landmarks = []
    with open('./data/detected.txt') as fp:
        for line in fp:
            pos = np.array([float(x) for x in line.rstrip().split(',')])
            landmarks.append(pos)

    start = time.time()
    projected_list, all_info = prj.projectLandmarksToMesh(mesh.vertices,
                                                          mesh.faces,
                                                          np.asarray(
                                                              landmarks),
                                                          fx, fy, cx, cy,
                                                          c2w_R, c2w_t)
    elapsed_time = time.time() - start
    print("elapsed_time:{0}".format(elapsed_time) + "[sec]")
    with open('intersections.json', 'w') as fp:
        json.dump(projected_list, fp, indent=4)

    projected_pos_list = [x['pos'] for x in projected_list if x is not None]
    writeMeshAsPly('projected.ply', projected_pos_list, [])


if __name__ == '__main__':
    main()
