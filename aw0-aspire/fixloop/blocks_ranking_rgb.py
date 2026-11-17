import numpy as np
from scipy import ndimage
from scipy.spatial.transform import Rotation as Rot

LIFT = 0.12          # robot property: carry clearance
Z_APPROACH = 0.10    # robot property: vertical approach standoff
GAP = 0.06           # robot property: open-jaw clearance between neighbouring blocks
OBJ_FOOT_MAX = 0.075  # block footprint bound (blocks are 3-5 cm cubes)
OBJ_TOP_MAX = 0.055   # block height bound above table
REACH_TOL = 0.03       # robot property: planner end-pose error above this means the wrist did not get there
ORDER = ["red", "green", "blue"]     # left-to-right order the task asks for


def world_cloud():
    cam = get_observation()["head_camera"]
    rgb, depth, K, T = cam["images"]["rgb"], cam["images"]["depth"], cam["intrinsics"], cam["pose_mat"]
    H, W = depth.shape
    pix = np.stack([*np.meshgrid(np.arange(W), np.arange(H)), np.ones((H, W))], -1).reshape(-1, 3).T
    cp = (np.linalg.inv(K) @ pix) * depth.ravel()
    wf = (T @ np.vstack([cp, np.ones(H * W)]))[:3].T.reshape(H, W, 3)
    wf[depth <= 0] = np.nan
    return rgb.astype(float), wf


def color_mask(rgb, c):
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    if c == "red":
        return (r > 90) & (r > g + 40) & (r > b + 40)
    if c == "green":
        return (g > 70) & (g > r + 30) & (g > b + 30)
    return (b > 90) & (b > r + 40) & (b > g + 40)


def perceive(c):
    """-> dict(center, top_pts, half): block of colour c from head RGB-D"""
    rgb, wf = world_cloud()
    ws = (np.abs(wf[..., 0]) < 0.45) & (wf[..., 1] > -0.35) & (wf[..., 1] < 0.25)
    anyc = color_mask(rgb, "red") | color_mask(rgb, "green") | color_mask(rgb, "blue")
    table_z = float(np.nanmedian(wf[ws & ~anyc][:, 2]))
    z = wf[..., 2]
    m = color_mask(rgb, c) & ws & (z > table_z + 0.01) & (z < table_z + 0.07)
    lbl, n = ndimage.label(m)
    best = None
    for i in range(1, n + 1):
        w = wf[lbl == i]
        ext = w[:, :2].max(0) - w[:, :2].min(0)
        top_i = float(np.percentile(w[:, 2], 90)) - table_z
        if len(w) >= 20 and ext.max() < OBJ_FOOT_MAX and 0.02 < top_i < OBJ_TOP_MAX and (best is None or len(w) > len(best)):
            best = w
    if best is None:
        raise RuntimeError(f"no {c} block seen")
    top = float(np.percentile(best[:, 2], 90))
    half = (top - table_z) / 2.0
    face = best[best[:, 2] > top - 0.006]
    cen = np.array([face[:, 0].mean(), face[:, 1].mean(), table_z + half])
    print("PERCEIVE", c, np.round(cen, 4).tolist(), "half", round(half, 4), "table_z", round(table_z, 4), "npts", len(best))
    return {"center": cen, "pts": best, "half": half}


def grasp_for(info, hand):
    obb = get_oriented_bounding_box_from_3d_points(info["pts"])
    gp, gq = get_top_down_grasp_from_obb(obb, hand=hand, points=info["pts"])
    g = np.array([info["center"][0], info["center"][1], info["center"][2]])
    print("GRASP api", np.round(gp, 4).tolist(), "used", np.round(g, 4).tolist())
    return g, np.asarray(gq, float)


def rotz_q(q, deg):
    r = Rot.from_euler("z", deg, degrees=True) * Rot.from_quat([q[1], q[2], q[3], q[0]])
    x = r.as_quat()
    return np.array([x[3], x[0], x[1], x[2]])


def move_block(c, info, slot_xy, last_hand):
    """pick block of colour c with the arm on its side and place it at slot_xy; -> hand used or None"""
    hand = "left" if info["center"][0] < 0 else "right"
    if last_hand is not None and last_hand != hand:
        goto_home_joint_position(last_hand)
    for attempt in range(2):
        g, q = grasp_for(info, hand)
        ok = False
        for dyaw in (0, 90, -90):
            qq = rotz_q(q, dyaw)
            for za in (Z_APPROACH, 0.05, 0.0):
                if goto_pose(hand, g, qq, z_approach=za):
                    ee = np.asarray(get_observation()[f"{hand}_endpose"])[:3]
                    if np.linalg.norm(ee[:2] - g[:2]) < REACH_TOL:
                        ok = True
                        break
                    print("GRASP short", c, dyaw, za, np.round(ee, 3).tolist())
            if ok:
                q = qq
                break
        if not ok:
            print("GRASP unreachable", c)
            return last_hand
        close_gripper(hand)
        ap = get_gripper_aperture(hand)
        print("aperture", c, round(ap, 3))
        if ap > 0.1:
            break
        open_gripper(hand)
        goto_pose(hand, g + np.array([0, 0, LIFT]), q)
        info = perceive(c)
    up = g + np.array([0, 0, LIFT])
    goto_pose(hand, up, q)
    tgt = np.array([slot_xy[0], slot_xy[1], g[2] + 0.004])
    placed = False
    for za in (Z_APPROACH, 0.05, 0.0):
        if goto_pose(hand, tgt, q, z_approach=za):
            placed = True
            break
    if not placed:
        print("PLACE failed", c)
    open_gripper(hand)
    goto_pose(hand, tgt + np.array([0, 0, LIFT]), q)
    return hand


# [setup]
infos = {c: perceive(c) for c in ORDER}
half = max(i["half"] for i in infos.values())
spacing = float(np.clip(2 * half + GAP, 0.09, 0.12))
spawn_y = float(np.mean([i["center"][1] for i in infos.values()]))
row_y = float(np.clip(spawn_y - 0.17, -0.20, -0.10))     # in front of the spawn band, inside both-arm reach
slots = {c: np.array([(k - 1) * spacing, row_y]) for k, c in enumerate(ORDER)}
print("SLOTS", {c: np.round(s, 3).tolist() for c, s in slots.items()}, "spacing", round(spacing, 3))

# [stage each] pick and place the {c} block
last_hand = None
for c in ORDER:
    last_hand = move_block(c, infos[c], slots[c], last_hand)

