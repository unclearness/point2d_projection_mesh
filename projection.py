import numpy as np

# An implementation of Möller-Trumbore algorithm
# https://www.scratchapixel.com/lessons/3d-basic-rendering/ray-tracing-rendering-a-triangle/moller-trumbore-ray-triangle-intersection
# https://pheema.hatenablog.jp/entry/ray-triangle-intersection#%E4%B8%89%E8%A7%92%E5%BD%A2%E3%81%AE%E5%86%85%E9%83%A8%E3%81%AB%E5%AD%98%E5%9C%A8%E3%81%99%E3%82%8B%E7%82%B9%E3%81%AE%E8%A1%A8%E7%8F%BE


def intersect(origin,  # [B, 3]
              ray,  # [B, 3]
              verts,  # [V, 3]
              faces,  # [F, 3]
              kEpsilon=1e-6,
              standard_barycentric=True,
              vert_normals=None,  # [V, 3]
              face_normals=None,  # [F, 3]
              cull_th=np.pi
              ):

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

    use_vert_normals = False
    use_face_normals = False
    if vert_normals is not None and verts.shape == vert_normals.shape:
        use_vert_normals = True
    if face_normals is not None and faces.shape == face_normals.shape:
        use_face_normals = True

    def interp_func_standard(u, v, v0, v1, v2):
        return u * v0 + v * v1 + (1 - u - v) * v2

    def interp_func_basis(u, v, v0, v1, v2):
        return u * (v1 - v0) + v * (v2 - v0) + v0

    interp_func = interp_func_basis
    if standard_barycentric:
        interp_func = interp_func_standard

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

        result_added = False
        for j in range(n):
            # Get the closest point
            t_ = t[i][indices[i]][j]
            u_ = u[i][indices[i]][j]
            v_ = v[i][indices[i]][j]
            fid = indices[i][j]

            if standard_barycentric:
                # Convert from edge basis vectors to triangle area ratios
                # Original: u * (v1 - v0) + v * (v2 - v0) + v0
                # Converted: u * v0 + v * v1 + (1 - u - v) * v2
                w_ = 1 - u_ - v_
                v_ = u_
                u_ = w_

            # Check culling
            if use_face_normals:
                fn = face_normals[fid]
                if np.arccos(ray[i].dot(fn)) > cull_th:
                    continue
            if use_vert_normals:
                vn = interp_func(
                    u_, v_,
                    vert_normals[fid], vert_normals[fid], vert_normals[fid])
                if np.arccos(ray[i].dot(vn)) > cull_th:
                    continue

            pos = org_origin[i] + org_ray[i] * t_
            # assert(
            #    (np.abs(interp_func(u_, v_, v0[fid], v1[fid], v2[fid]) - pos)
            #     < 1e-4).all())

            result = {'t': t_, 'u': u_, 'v': v_,
                      'fid': int(fid), 'pos': pos.tolist()}
            results.append(result)
            result_added = True
            break
        if not result_added:
            results.append(None)

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
