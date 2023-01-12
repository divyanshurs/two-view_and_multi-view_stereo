import numpy as np
import cv2
import sys

EPS = 1e-8


def backproject_corners(K, width, height, depth, Rt):
    """
    Backproject 4 corner points in image plane to the imaginary depth plane

    Input:
        K -- camera intrinsics calibration matrix
        Rt -- 3 x 4 camera extrinsics calibration matrix
        width -- width of the image
        heigh -- height of the image
        depth -- depth value of the imaginary plane
    Output:
        points -- 2 x 2 x 3 array of 3D coordinates of backprojected points
    """

    points = np.array((
        (0, 0, 1),
        (width, 0, 1),
        (0, height, 1),
        (width, height, 1),
    ), dtype=np.float32).reshape(2, 2, 3)
    
    ret_points = np.zeros((2,2,3))

    """ YOUR CODE HERE
    """
    K_inv = np.linalg.inv(K)
    T = np.vstack((Rt, [0,0,0,1]))
    T_inv = np.linalg.inv(T)
    for s in range(points.shape[0]):
        for p in range(points.shape[1]):
            p_int = points[s,p, :]
            p_ray = K_inv@p_int
            p_ray = depth*p_ray
            aug_p = np.hstack((p_ray, 1))
            p_world = T_inv@aug_p
            p_world = p_world[:3]
            ret_points[s,p,:] = p_world
             
    """ END YOUR CODE
    """
    return ret_points

def project_points(K, Rt, points):
    """
    Project 3D points into a calibrated camera.
    Input:
        K -- camera intrinsics calibration matrix
        Rt -- 3 x 4 camera extrinsics calibration matrix
        points -- points_height x points_width x 3 array of 3D points
    Output:
        projections -- points_height x points_width x 2 array of 2D projections
    """
    """ YOUR CODE HERE
    """
    T = np.vstack((Rt, [0,0,0,1]))
    p_ret = np.zeros((points.shape[0], points.shape[1], 2))

    for s in range(points.shape[0]):
        for p in range(points.shape[1]):
            p_int = points[s,p, :]
            aug_p = np.hstack((p_int, 1))
            p_cam = T@aug_p
            p_project = K@p_cam[:3]
            p_project = p_project[:2]/p_project[2]
            p_ret[s, p,:] = p_project

    """ END YOUR CODE
    """
    return p_ret

def warp_neighbor_to_ref(backproject_fn, project_fn, depth, neighbor_rgb, K_ref, Rt_ref, K_neighbor, Rt_neighbor):
    """ 
    Warp the neighbor view into the reference view 
    via the homography induced by the imaginary depth plane at the specified depth

    Make use of the functions you've implemented in the same file (which are passed in as arguments):
    - backproject_corners
    - project_points

    Also make use of the cv2 functions:
    - cv2.findHomography
    - cv2.warpPerspective

    Input:
        backproject_fn -- backproject_corners function
        project_fn -- project_points function
        depth -- scalar value of the depth at the imaginary depth plane
        neighbor_rgb -- height x width x 3 array of neighbor rgb image
        K_ref -- 3 x 3 camera intrinsics calibration matrix of reference view
        Rt_ref -- 3 x 4 camera extrinsics calibration matrix of reference view
        K_neighbor -- 3 x 3 camera intrinsics calibration matrix of neighbor view
        Rt_neighbor -- 3 x 4 camera extrinsics calibration matrix of neighbor view
    Output:
        warped_neighbor -- height x width x 3 array of the warped neighbor RGB image
    """

    height, width = neighbor_rgb.shape[:2]

    points = np.array((
        (0, 0),
        (width, 0),
        (0, height),
        (width, height),
    ))

    """ YOUR CODE HERE
    """
    points_pro = backproject_fn(K_ref,width, height, depth, Rt_ref) #project neighbour to given depth

    final_points = project_fn(K_neighbor, Rt_neighbor, points_pro) #project depth points back into ref image
    final_points = np.reshape(final_points, (4,2))
    
    #points_pro = points_pro[:,:,:2]
    # points_pro = np.reshape(points_pro, (4,3))
    # points_pro = points_pro/points_pro[:,-1].reshape((4,1))  
    # points_pro = points_pro[:,:2]

    H, _ = cv2.findHomography(points, final_points, cv2.RANSAC)
    warped_neighbor = cv2.warpPerspective(neighbor_rgb, np.linalg.inv(H), (width, height))
    """ END YOUR CODE
    """
    return warped_neighbor


def zncc_kernel_2D(src, dst):
    """ 
    Compute the cost map between src and dst patchified images via the ZNCC metric
    
    IMPORTANT NOTES:
    - Treat each RGB channel separately but sum the 3 different zncc scores at each pixel

    - When normalizing by the standard deviation, add the provided small epsilon value, 
    EPS which is included in this file, to both sigma_src and sigma_dst to avoid divide-by-zero issues

    Input:
        src -- height x width x K**2 x 3
        dst -- height x width x K**2 x 3
    Output:
        zncc -- height x width array of zncc metric computed at each pixel
    """
    assert src.ndim == 4 and dst.ndim == 4
    assert src.shape[:] == dst.shape[:]

    """ YOUR CODE HERE
    """

    src_mean = np.mean(src, axis=2).reshape((src.shape[0], src.shape[1], 1, src.shape[3]))
    src_std = np.std(src, axis=2).reshape((src.shape[0], src.shape[1], 1, src.shape[3]))
    dst_mean = np.mean(dst, axis=2).reshape((dst.shape[0], dst.shape[1], 1, dst.shape[3]))
    dst_std = np.std(dst, axis=2).reshape((dst.shape[0], dst.shape[1], 1, dst.shape[3]))
    src_mean_val = src- src_mean 
    dst_mean_val = dst- dst_mean

    num = src_mean_val*dst_mean_val
    den = src_std*dst_std
    den[np.where(den==0)] = EPS

    val = num/den
    zncc = np.sum(val, axis=-1).sum(axis=-1)

    """ END YOUR CODE
    """

    return zncc  # height x width


def backproject(dep_map, K):
    """ 
    Backproject image points to 3D coordinates wrt the camera frame according to the depth map

    Input:
        K -- camera intrinsics calibration matrix
        dep_map -- height x width array of depth values
    Output:
        points -- height x width x 3 array of 3D coordinates of backprojected points
    """
    _u, _v = np.meshgrid(np.arange(dep_map.shape[1]), np.arange(dep_map.shape[0]))

    """ YOUR CODE HERe
    """
    
    bla = np.array((_u.flatten(), _v.flatten(),np.ones((_u.shape[0], _u.shape[1])).flatten()))
    d = bla*dep_map.flatten()
    s = np.linalg.inv(K)@d



    xyz_cam = np.reshape(s.T, (_u.shape[0], _u.shape[1], 3))
     
    """ END YOUR CODE
    """
    return xyz_cam

