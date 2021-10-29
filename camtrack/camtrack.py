#! /usr/bin/env python3

__all__ = [
    'track_and_calc_colors'
]

from typing import List, Optional, Tuple

import numpy as np
import cv2
from sklearn.preprocessing import normalize
import _camtrack
import _corners

from corners import CornerStorage
from data3d import CameraParameters, PointCloud, Pose
import frameseq
from _camtrack import (
    PointCloudBuilder,
    create_cli,
    calc_point_cloud_colors,
    pose_to_view_mat3x4,
    to_opencv_camera_mat3x3,
    view_mat3x4_to_pose
)


def track_and_calc_colors(camera_parameters: CameraParameters,
                          corner_storage: CornerStorage,
                          frame_sequence_path: str,
                          known_view_1: Optional[Tuple[int, Pose]] = None,
                          known_view_2: Optional[Tuple[int, Pose]] = None) \
        -> Tuple[List[Pose], PointCloud]:

    if known_view_1 is None or known_view_2 is None:
        raise NotImplementedError()

    rgb_sequence = frameseq.read_rgb_f32(frame_sequence_path)
    intrinsic_mat = to_opencv_camera_mat3x3(
        camera_parameters,
        rgb_sequence[0].shape[0]
    )

    # TODO: implement
    frame_count = len(corner_storage)
    triangulation_parameters = _camtrack.TriangulationParameters(2, 1, 1)
    scope = max(int(abs(known_view_1[0] - known_view_2[0]) / 3), int(frame_count * 0.05))
    dist_coefs = np.array([0, 0, 0, 0, 0], dtype=float)

    frames_not_checked = np.ones(frame_count)
    frames_not_checked[known_view_1[0]] = 0
    frames_not_checked[known_view_2[0]] = 0
    corners_view1 = corner_storage[known_view_1[0]]
    corners_view1 = corner_storage[known_view_1[0]]
    corners_view2 = corner_storage[known_view_2[0]]
    correspondence = _camtrack.build_correspondences(corners_view1, corners_view2)
    points_3d, correspondence_ids, mcos = _camtrack.triangulate_correspondences(correspondence,
                                                                                pose_to_view_mat3x4(known_view_1[1]),
                                                                                pose_to_view_mat3x4(known_view_2[1]),
                                                                                intrinsic_mat,
                                                                                _camtrack.TriangulationParameters(5,
                                                                                                                  0,
                                                                                                                  0))
    view_mats = [pose_to_view_mat3x4(known_view_1[1])] * frame_count
    view_mats[known_view_1[0]] = pose_to_view_mat3x4(known_view_1[1])
    view_mats[known_view_2[0]] = pose_to_view_mat3x4(known_view_2[1])
    count_checked = 2
    next_i = [min(known_view_1[0], known_view_2[0]) - 1,
              min(known_view_1[0], known_view_2[0]) + 1,
              max(known_view_1[0], known_view_2[0]) - 1,
              max(known_view_1[0], known_view_2[0]) + 1]
    next_rvecs = [None, None, None, None]
    next_tvecs = [None, None, None, None]
    next_i_step = [-1, 1, -1, 1]
    next_i_counter = 0
    while count_checked != frame_count:
        i = next_i[next_i_counter % 4]
        if i < 0 or i >= frame_count:
            next_i_counter += 1
            continue
        if frames_not_checked[i] == 0:
            next_i_counter += 1
            continue
        # CALC NEW POSE #
        cur_corners = corner_storage[i]
        good_ids_bool = np.array([True if a in correspondence_ids else False for a in cur_corners.ids])
        good_corners = _corners.filter_frame_corners(cur_corners, good_ids_bool)
        good_points = []
        for j in range(len(correspondence_ids)):
            if correspondence_ids[j] in good_corners.ids:
                good_points.append([correspondence_ids[j], points_3d[j]])
        good_points = sorted(good_points, key=lambda x: x[0])
        good_points = [x[1] for x in good_points]
        rvec = next_rvecs[next_i_counter % 4]
        tvec = next_tvecs[next_i_counter % 4]
        success, rvec, tvec, inliers = cv2.solvePnPRansac(np.array(good_points), good_corners.points, intrinsic_mat,
                                                          dist_coefs, rvec, tvec,
                                                          useExtrinsicGuess=1,
                                                          flags=cv2.SOLVEPNP_ITERATIVE,
                                                          reprojectionError=2,
                                                          iterationsCount=5000,
                                                          confidence=0.8)
        next_rvecs[next_i_counter % 4] = rvec
        next_tvecs[next_i_counter % 4] = tvec
        if not success:
            continue
        view_mat = _camtrack.rodrigues_and_translation_to_view_mat3x4(rvec, tvec)
        view_mats[i] = view_mat

        # CALC NEW POINTS #

        camera_center = _camtrack.to_camera_center(view_mat)
        max_distance = 0
        max_idx = 0
        min_angle = 1
        for j in range(max(0, i - scope),
                       min(frame_count, i + scope)):
            if frames_not_checked[j]:
                continue
            camera_center_new = _camtrack.to_camera_center(view_mats[j])
            distance = np.linalg.norm(camera_center_new - camera_center)
            vecs_1 = normalize(camera_center - points_3d)
            vecs_2 = normalize(camera_center_new - points_3d)
            angle_cos = np.mean(np.abs(np.einsum('ij,ij->i', vecs_1, vecs_2)))
            if distance >= max_distance and angle_cos <= min_angle:
                min_angle = angle_cos
                max_distance = distance
                max_idx = j
        corners_view1 = cur_corners
        corners_view2 = corner_storage[max_idx]
        correspondence = _camtrack.build_correspondences(corners_view2, corners_view1)
        new_points_3d, new_correspondence_ids, mcos = \
            _camtrack.triangulate_correspondences(correspondence,
                                                  view_mats[max_idx],
                                                  view_mat,
                                                  intrinsic_mat,
                                                  triangulation_parameters)

        # ADDING NEW POINTS TO CLOUD #
        new_points_count = 0
        edited_points = 0
        for j in range(len(new_correspondence_ids)):
            if new_correspondence_ids[j] not in correspondence_ids:
                correspondence_ids = np.append(correspondence_ids, new_correspondence_ids[j])
                points_3d = np.vstack([points_3d, new_points_3d[j]])
                new_points_count += 1
            else:
                k1 = np.where(correspondence_ids == new_correspondence_ids[j])[0][0]
                # p = max_idx
                p = known_view_1[0]
                corners = corner_storage[p]
                try:
                    k2 = np.where(corners.ids == new_correspondence_ids[j])[0][0]
                except IndexError:
                    continue
                err1 = _camtrack.compute_reprojection_errors(np.array([points_3d[k1], np.array([0, 0, 0])]),
                                                             np.array([corners.points[k2], np.array([0, 0])]),
                                                             intrinsic_mat @ view_mats[p])[0]
                err2 = _camtrack.compute_reprojection_errors(np.array([new_points_3d[j], np.array([0, 0, 0])]),
                                                             np.array([corners.points[k2], np.array([0, 0])]),
                                                             intrinsic_mat @ view_mats[p])[0]
                if err1 > err2:
                    points_3d[k1] = new_points_3d[j]
                    edited_points += 1

        print('Checked - {}, Current frame - {}, inlier count - {}, new triangulated points - {}, edited points - {}, cloud size - {}'
              .format(count_checked, i, len(inliers), new_points_count, edited_points, len(points_3d)))
        frames_not_checked[i] = 0
        next_i[next_i_counter % 4] += next_i_step[next_i_counter % 4]
        next_i_counter += 1
        count_checked += 1

    # RECOUNT #

    dists = []
    camera_center_1 = _camtrack.to_camera_center(view_mats[0])
    for i in range(1, len(view_mats)):
        camera_center_2 = _camtrack.to_camera_center(view_mats[i - 1])
        distance = np.linalg.norm(camera_center_2 - camera_center_1)
        dists.append(distance)
        camera_center_1 = camera_center_2
    speed = sum(dists) / len(view_mats)
    print(speed)

    if speed > 6:
        meth = cv2.SOLVEPNP_P3P
    else:
        meth = cv2.SOLVEPNP_ITERATIVE

    def recount():
        rv = None
        tv = None
        for new_i in range(frame_count):
            #if new_i == known_view_1[0] or new_i == known_view_2[0]:
            #    continue
            new_cur_corners = corner_storage[new_i]
            new_good_ids_bool = np.array([True if a in correspondence_ids else False for a in new_cur_corners.ids])
            new_good_corners = _corners.filter_frame_corners(new_cur_corners, new_good_ids_bool)
            new_good_points = []
            for j in range(len(correspondence_ids)):
                if correspondence_ids[j] in new_good_corners.ids:
                    new_good_points.append([correspondence_ids[j], points_3d[j]])
            new_good_points = sorted(new_good_points, key=lambda x: x[0])
            new_good_points = [x[1] for x in new_good_points]
            succ, rv, tv, inl = cv2.solvePnPRansac(np.array(new_good_points), new_good_corners.points,
                                                   intrinsic_mat,
                                                   dist_coefs, rv, tv,
                                                   useExtrinsicGuess=1,
                                                   flags=meth,
                                                   reprojectionError=1,
                                                   iterationsCount=5000,
                                                   confidence=0.6)
            if not succ:
                continue
            view_mats[new_i] = _camtrack.rodrigues_and_translation_to_view_mat3x4(rv, tv)
            print('Recount of {} position, inliers count - {}'.format(new_i, len(inl)))

    recount()

    point_cloud_builder = PointCloudBuilder(correspondence_ids, points_3d)
    calc_point_cloud_colors(
        point_cloud_builder,
        rgb_sequence,
        view_mats,
        intrinsic_mat,
        corner_storage,
        5.0
    )
    point_cloud = point_cloud_builder.build_point_cloud()
    poses = list(map(view_mat3x4_to_pose, view_mats))
    return poses, point_cloud


if __name__ == '__main__':
    # pylint:disable=no-value-for-parameter
    create_cli(track_and_calc_colors)()
