import json
import multiprocessing
import time
import numpy as np
import trimesh


# An implementation of Möller-Trumbore algorithm
# https://www.scratchapixel.com/lessons/3d-basic-rendering/ray-tracing-rendering-a-triangle/moller-trumbore-ray-triangle-intersection
# https://pheema.hatenablog.jp/entry/ray-triangle-intersection#%E4%B8%89%E8%A7%92%E5%BD%A2%E3%81%AE%E5%86%85%E9%83%A8%E3%81%AB%E5%AD%98%E5%9C%A8%E3%81%99%E3%82%8B%E7%82%B9%E3%81%AE%E8%A1%A8%E7%8F%BE
def intersect(origin,  # [B, 3]
              ray,  # [B, 3]
              verts,  # [V, 3]
              faces,  # [F, 3]
              kEpsilon=1e-6, standard_barycentric=True):

    fetched = verts[faces]  # [F, 3, 3]
    v0 = fetched[:, 0]    # [F, 3]
    v1 = fetched[:, 1]
    v2 = fetched[:, 2]

    e1 = v1 - v0
    e2 = v2 - v0

    F = len(faces)
    B = len(origin)

    org_origin, org_ray = origin, ray
    ray = np.repeat(ray[:, None], F, axis=1)  # [B, F, 3]
    origin = np.repeat(origin[:, None], F, axis=1)

    alpha = np.cross(ray, e2)  # [B, F, 3]

    det = np.einsum('ft,bft->bf', e1, alpha)

    # if det == 0, ray is parallel to triangle
    # Let's ignore this numerically unstable case
    small_det_mask = (-kEpsilon < det) & (det < kEpsilon)  # [B, F]

    invDet = 1.0 / det   # [B, F]
    r = origin - v0   # [B, F, 3]

    # u, v and t are defined as follows:
    # origin + ray * t == (1−u−v)*v0 + u*v1 +v*v2
    # Must be 0 <= u <= 1
    u = np.einsum('bft,bft->bf', alpha, r) * invDet  # [B, F]
    u_invalid_mask = (u < 0.0) | (u > 1.0)   # [B, F]

    beta = np.cross(r, e1)   # [B, F, 3]

    # Must be 0 <= v <= 1 and u + v <= 1
    # Thus, 0 <= v <= 1 - u
    v = np.einsum('bft,bft->bf', ray, beta) * invDet  # [B, F]
    v_invalid_mask = (v < 0.0) | (u + v > 1.0)

    # The triangle is in the front of ray
    # So, 0 <= t
    t = np.einsum('ft,bft->bf', e2, beta) * invDet
    t_invalid_mask = t < 0.0

    # [B, F]
    invalid_mask = small_det_mask | u_invalid_mask \
        | v_invalid_mask | t_invalid_mask

    # [B]
    N = F - invalid_mask.sum(axis=-1)
    t[invalid_mask] = np.inf

    # [B, F]
    indices = np.argsort(t, axis=-1)
    results = []
    # #intersections N[i] varies per batch B[i],
    # which prevents from batch operation.
    # Let's use loop
    for i in range(B):
        n = N[i]
        if n < 1:
            results.append(None)
            continue
        t_ = t[i][indices[i]][:n]
        u_ = u[i][indices[i]][:n]
        v_ = v[i][indices[i]][:n]
        fid = indices[i][:n]

        idx = np.argmin(t_)

        if standard_barycentric:
            # Convert from edge vector to triangle area ratio
            w_ = 1 - u_ - v_
            v_ = u_
            u_ = w_

        pos = org_origin[i] + org_ray[i] * t_[idx]
        result = {'t': t_[idx], 'u': u_[idx], 'v': v_[
            idx], 'fid': int(fid[idx]), 'pos': pos.tolist()}
        results.append(result)

    return results, (t, u, v, indices, N)


def intersectNaiveBase(origin, ray, v0, v1, v2, kEpsilon=1e-6,
                       standard_barycentric=True):
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

    if standard_barycentric:
        w = 1 - u - v
        v = u
        u = w

    return (t, u, v)


def intersectNaive(origin, ray, verts, faces, kEpsilon=1e-6,
                   standard_barycentric=True):
    results = []
    all_info = None
    B = len(origin)
    for i in range(B):
        result_per_batch = []
        for fid, face in enumerate(faces):
            v0 = verts[face[0]]
            v1 = verts[face[1]]
            v2 = verts[face[2]]
            ret = intersectNaiveBase(origin[i], ray[i], v0, v1, v2,
                                     kEpsilon, standard_barycentric)
            if ret is None:
                continue
            t, u, v = ret
            pos = origin[i] + ray[i] * t
            result_ = {'t': t, 'u': u, 'v': v, 'fid': fid, 'pos': pos.tolist()}
            result_per_batch.append(result_)
        if len(result_per_batch) == 0:
            result = None
        else:
            result = result_per_batch[0]
            for i in range(1, len(result_per_batch)):
                if result_per_batch[i]['t'] < result['t']:
                    result = result_per_batch[i]
        results.append(result)
    return results, all_info


def intersectMesh(origin, ray, verts, faces, kEpsilon=1e-6, mode="batch_all"):
    if mode == "batch_geom":
        results = []
        for i in range(len(origin)):
            result = intersect(origin[i][None, ], ray[i]
                               [None, ], verts, faces, kEpsilon)
            results.append(result[0][0])
        results = (results, None)
    elif mode == "batch_all":
        results = intersect(origin, ray, verts, faces, kEpsilon)
    else:
        results = intersectNaive(origin, ray, verts, faces, kEpsilon)
    return results


def intrinsic2ray(pos, fx, fy, cx, cy):
    ray = np.zeros((pos.shape[0], 3))
    ray[..., 0] = (pos[..., 0] - cx) / fx
    ray[..., 1] = (pos[..., 1] - cy) / fy
    ray[..., 2] = 1.0
    ray = ray / np.linalg.norm(ray, axis=-1, keepdims=True)
    return ray


def projectLandmarksToMesh(verts, faces, pos, fx, fy, cx, cy, c2w_R, c2w_t):
    origin = c2w_t
    cam_ray = intrinsic2ray(pos, fx, fy, cx, cy)
    wld_ray = np.einsum('mn,ln->lm', c2w_R, cam_ray)
    origin = origin[None, ].repeat(len(cam_ray), axis=0)
    results = intersectMesh(origin, wld_ray, verts, faces)
    return results


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
    projected_list, all_info = projectLandmarksToMesh(mesh.vertices,
                                                      mesh.faces,
                                                      np.asarray(landmarks),
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
