import numpy as np
import json
import trimesh
from typing import NamedTuple


def intersect(origin, ray, v0, v1, v2, kEpsilon=1e-6):
    e1 = v1 - v0
    e2 = v2 - v0

    alpha = np.cross(ray, e2)
    det = e1.dot(alpha)

    # if det == 0, ray is parallel to triangle
    # Let's ignore this numerically unstable case
    if -kEpsilon < det and det < kEpsilon:
        return None

    invDet = 1.0 / det
    r = origin - v0

    # u, v and t are defined as follows:
    # origin + ray * t == (1−u−v)*v0 + u*v1 +v*v2
    # Must be 0 <= u <= 1
    u = alpha.dot(r) * invDet
    if u < 0.0 or u > 1.0:
        return None

    beta = np.cross(r, e1)

    # Must be 0 <= v <= 1 and u + v <= 1
    # Thus, 0 <= v <= 1 - u
    v = ray.dot(beta) * invDet
    if v < 0.0 or u + v > 1.0:
        return None

    # The triangle is in the front of ray
    # So, 0 <= t
    t = e2.dot(beta) * invDet
    if t < 0.0:
        return None

    return (t, u, v)


# class InteresectResult:
#     def __init__(self, t, u, v, fid, pos):
#         self.t = t
#         self.u = u
#         self.v = v
#         self.fid = fid
#         self.pos = pos


def json_serializable(cls):
    def as_dict(self):
        yield {name: value for name, value in zip(
            self._fields,
            iter(super(cls, self).__iter__()))}
    cls.__iter__ = as_dict
    return cls


@json_serializable
class InteresectResult(NamedTuple):
    t: float
    u: float
    v: float
    fid: int
    pos: list


def intersectMesh(origin, ray, verts, faces, kEpsilon=1e-6):
    results = []
    # TODO: Batch version
    for fid, face in enumerate(faces):
        v0 = verts[face[0]]
        v1 = verts[face[1]]
        v2 = verts[face[2]]
        ret = intersect(origin, ray, v0, v1, v2, kEpsilon)
        if ret is None:
            continue
        t, u, v = ret
        pos = origin + ray * t
        result = InteresectResult(t, u, v, fid, pos.tolist())
        results.append(result)
    return results


def intrinsic2ray(pos, fx, fy, cx, cy):
    ray = np.zeros((3))
    ray[0] = (pos[0] - cx) / fx
    ray[1] = (pos[1] - cy) / fy
    ray[2] = 1.0
    ray = ray / np.linalg.norm(ray)
    return ray


def projectLandmarkToMesh(verts, faces, pos, fx, fy, cx, cy, c2w_R, c2w_t):
    origin = c2w_t
    cam_ray = intrinsic2ray(pos, fx, fy, cx, cy)
    wld_ray = c2w_R.dot(cam_ray)
    results = intersectMesh(origin, wld_ray, verts, faces)
    if len(results) < 1:
        return None
    if len(results) == 1:
        return results[0]
    closest = results[0]
    for res in results[1:]:
        if res.t < closest.t:
            closest = res
    return closest


def projectLandmarksToMesh(verts, faces, landmarks,
                           fx, fy, cx, cy, c2w_R, c2w_t):
    projected_list = []
    for lmk in landmarks:
        closest = projectLandmarkToMesh(verts, faces, lmk, fx, fy, cx, cy,
                                        c2w_R, c2w_t)
        print(lmk, closest)
        projected_list.append(closest)
    return projected_list


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
    mesh = trimesh.load('./data/max-planck_10k.obj')

    # Load camera param
    with open('./data/camera_param.json') as fp:
        camera_param = json.load(fp)
    K = camera_param['K']
    fx = K[0][0]
    fy = K[1][1]
    cx = K[0][2]
    cy = K[1][2]
    # print(fx, fy, cx, cy)
    w2c_R = np.array(camera_param['R_world2cv'])
    w2c_t = np.array(camera_param['T_world2cv'])
    # print(w2c_R, w2c_t)
    c2w_R = w2c_R.T
    c2w_t = -1 * c2w_R.dot(w2c_t)
    # print(c2w_R, c2w_t)

    # Load landmarks
    landmarks = []
    with open('./data/detected.txt') as fp:
        for line in fp:
            pos = np.array([float(x) for x in line.rstrip().split(',')])
            landmarks.append(pos)
    # print(landmarks)

    projected_list = projectLandmarksToMesh(mesh.vertices, mesh.faces,
                                            landmarks,
                                            fx, fy, cx, cy, c2w_R, c2w_t)

    with open('intersections.json', 'w') as fp:
        json.dump(projected_list, fp, indent=4)

    projected_pos_list = [x.pos for x in projected_list if x is not None]
    writeMeshAsPly('projected.ply', projected_pos_list, [])


if __name__ == '__main__':
    main()
