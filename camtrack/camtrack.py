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
    rgb_sequence = frameseq.read_rgb_f32(frame_sequence_path)
    intrinsic_mat = to_opencv_camera_mat3x3(
        camera_parameters,
        rgb_sequence[0].shape[0]
    )

    # TODO: implement

    BASELINE = 0.9
    RECOUNT_REPROJECTION_ERROR = 0.9
    REPROJECTION_ERROR = 1.4
    REPR_ERROR = 3
    MIN_3D_POINTS = 10

    #BASELINE = 0.9
    #RECOUNT_REPROJECTION_ERROR = 0.9
    #REPROJECTION_ERROR = 1.4
    #REPR_ERROR = 1.4
    #MIN_3D_POINTS = 10

    frame_count = len(corner_storage)

    view_mats = [0] * frame_count

    frames_of_corner = {}
    for i, corners in enumerate(corner_storage):
        for id_in_list, j in enumerate(corners.ids.flatten()):
            if j not in frames_of_corner.keys():
                frames_of_corner[j] = [[i, id_in_list]]
            else:
                frames_of_corner[j].append([i, id_in_list])

    def triangulate(f_id_0, f_id_1, params=_camtrack.TriangulationParameters(2, 0.001, 0.0001)):
        return _camtrack.triangulate_correspondences(
            _camtrack.build_correspondences(corner_storage[f_id_0], corner_storage[f_id_1]),
            view_mats[f_id_0], view_mats[f_id_1], intrinsic_mat, params)

    def calc_views():
        print('initializing started')

        frame_steps = [9, 30, 40]
        def get_first_cam_pose(frame_0, frame_1):
            corrs = _camtrack.build_correspondences(corner_storage[frame_0], corner_storage[frame_1])
            essential_mat, _ = cv2.findEssentialMat(corrs.points_1, corrs.points_2, intrinsic_mat, cv2.RANSAC, 0.999, 2.0)

            pos_rvec1, pos_rvec2, pos_tvec = cv2.decomposeEssentialMat(essential_mat)
            pos_views = [[pos_rvec1, pos_tvec],
                         [pos_rvec2, pos_tvec],
                         [pos_rvec1, -pos_tvec],
                         [pos_rvec2, -pos_tvec]]
            best_R, best_t = pos_views[0]
            max_positive_z = 0
            for [R, t] in pos_views:
                view_mat0 = np.hstack((np.eye(3), np.zeros((3, 1))))
                view_mat1 = np.hstack((R, t.reshape((3, 1))))
                view_mats[frame_0] = view_mat0
                view_mats[frame_1] = view_mat1
                points3d, ids, mcos = triangulate(frame_0, frame_1)
                cur_positive_z = 0
                for pt in points3d:
                    pt = np.append(pt, np.zeros((1))).reshape((4, 1))
                    pt_transformed0 = (view_mat0 @ pt).flatten()
                    z0 = pt_transformed0[2]
                    pt_transformed1 = (view_mat1 @ pt).flatten()
                    z1 = pt_transformed1[2]
                    cur_positive_z += (z0 > 0.1) + (z1 > 0.1)
                if cur_positive_z > max_positive_z:
                    max_positive_z = cur_positive_z
                    best_R = R
                    best_t = t

            baseline = np.linalg.norm(best_t)

            view_mat0 = np.hstack((np.eye(3), np.zeros((3, 1))))
            view_mat1 = np.hstack((best_R, best_t.reshape((3, 1))))
            view_mats[frame_0] = view_mat0
            view_mats[frame_1] = view_mat1
            points3d, ids, median_cos = triangulate(frame_0, frame_1)
            points0 = []
            points1 = []
            for i in ids:
                for j in frames_of_corner[i]:
                    if j[0] == frame_0:
                        points0.append(j[1])
                    if j[0] == frame_1:
                        points1.append(j[1])
            points0 = np.array(points0)
            points1 = np.array(points1)
            sum_error = _camtrack.compute_reprojection_errors(points3d, corner_storage[frame_0].points[points0],
                                                              intrinsic_mat @ view_mat0) + _camtrack.compute_reprojection_errors(
                points3d,
                corner_storage[
                    frame_1].points[
                    points1],
                intrinsic_mat @ view_mat1)
            reprojection_error = np.mean(sum_error)
            points3d_count = len(points3d)

            if baseline < BASELINE or reprojection_error > REPROJECTION_ERROR or points3d_count < MIN_3D_POINTS:
                return False, baseline, reprojection_error, points3d_count, None, None
            else:
                return True, baseline, reprojection_error, points3d_count, best_R, best_t

        best_frame_step = 5
        ind = None
        for frame_step in frame_steps:
            for i in range(frame_count // 2):
                if i + frame_step < frame_count * 0.85:
                    succ, baseline, rep_err, points3d_cnt, Rx, tx = get_first_cam_pose(i, i + frame_step)
                    if succ:
                        ind = i
                        best_frame_step = frame_step
                        if frame_count > 100:
                            break

        frame_0 = ind
        frame_1 = ind + best_frame_step
        succ, baseline, rep_err, points3d_cnt, R, t = get_first_cam_pose(frame_0, frame_1)
        print('Done')
        view_mat0 = np.hstack((np.eye(3), np.zeros((3, 1))))
        view_mat1 = np.hstack((R, t.reshape((3, 1))))

        return [frame_0, _camtrack.view_mat3x4_to_pose(view_mat0)], \
               [frame_1, _camtrack.view_mat3x4_to_pose(view_mat1)]

    if known_view_1 is None or known_view_2 is None:
        known_view_1, known_view_2 = calc_views()

    triangulation_parameters = _camtrack.TriangulationParameters(1.5, 10, 0.00001)
    scope = max(1, int(abs(known_view_1[0] - known_view_2[0]) / 3), int(frame_count * 0.05))
    dist_coefs = np.array([0, 0, 0, 0, 0], dtype=float)

    frames_not_checked = np.ones(frame_count)
    frames_not_checked[known_view_1[0]] = 0
    frames_not_checked[known_view_2[0]] = 0
    corners_view1 = corner_storage[known_view_1[0]]
    corners_view2 = corner_storage[known_view_2[0]]
    correspondence = _camtrack.build_correspondences(corners_view1, corners_view2)
    points_3d, correspondence_ids, mcos = _camtrack.triangulate_correspondences(correspondence,
                                                                                pose_to_view_mat3x4(known_view_1[1]),
                                                                                pose_to_view_mat3x4(known_view_2[1]),
                                                                                intrinsic_mat,
                                                                                _camtrack.TriangulationParameters(5,
                                                                                                                  0.1,
                                                                                                                  0.001))
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

        conf = 0.99999
        repr = REPR_ERROR

        success, rv, tv, inliers = cv2.solvePnPRansac(np.array(good_points), good_corners.points, intrinsic_mat,
                                                      dist_coefs, rvec, tvec,
                                                      useExtrinsicGuess=1,
                                                      flags=cv2.SOLVEPNP_ITERATIVE,
                                                      reprojectionError=repr,
                                                      confidence=conf)
        next_rvecs[next_i_counter % 4] = rv
        next_tvecs[next_i_counter % 4] = tv
        while not success:
            conf -= 0.02
            repr += 0.1
            if conf <= 0:
                break
            success, rv, tv, inliers = cv2.solvePnPRansac(np.array(good_points), good_corners.points, intrinsic_mat,
                                                          dist_coefs, rvec, tvec,
                                                          useExtrinsicGuess=1,
                                                          flags=cv2.SOLVEPNP_ITERATIVE,
                                                          reprojectionError=repr,
                                                          confidence=conf)
            next_rvecs[next_i_counter % 4] = rv
            next_tvecs[next_i_counter % 4] = tv
        view_mat = _camtrack.rodrigues_and_translation_to_view_mat3x4(rv, tv)
        view_mats[i] = view_mat

        # CALC NEW POINTS #

        camera_center = _camtrack.to_camera_center(view_mat)
        max_distance = 0
        max_idx = 0
        for j in range(max(0, i - scope),
                       min(frame_count, i + scope)):
            if frames_not_checked[j]:
                continue
            camera_center_new = _camtrack.to_camera_center(view_mats[j])
            distance = np.linalg.norm(camera_center_new - camera_center)
            vecs_1 = normalize(camera_center - points_3d)
            vecs_2 = normalize(camera_center_new - points_3d)
            if distance >= max_distance:
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
                for p in [known_view_1[0], known_view_2[0]]:
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

        print(
            'Checked - {}, Current frame - {}, inlier count - {}, new triangulated points - {}, edited points - {}, cloud size - {}'
                .format(count_checked, i, len(inliers), new_points_count, edited_points, len(points_3d)))
        frames_not_checked[i] = 0
        next_i[next_i_counter % 4] += next_i_step[next_i_counter % 4]
        next_i_counter += 1
        count_checked += 1

    # RECOUNT #
    meth = cv2.SOLVEPNP_ITERATIVE

    def recount():
        rv = None
        tv = None
        for new_i in range(frame_count):
            new_cur_corners = corner_storage[new_i]
            new_good_ids_bool = np.array([True if a in correspondence_ids else False for a in new_cur_corners.ids])
            new_good_corners = _corners.filter_frame_corners(new_cur_corners, new_good_ids_bool)
            new_good_points = []
            for j in range(len(correspondence_ids)):
                if correspondence_ids[j] in new_good_corners.ids:
                    new_good_points.append([correspondence_ids[j], points_3d[j]])
            new_good_points = sorted(new_good_points, key=lambda x: x[0])
            new_good_points = [x[1] for x in new_good_points]

            conf = 0.99999
            repr = RECOUNT_REPROJECTION_ERROR

            succ, rvec, tvec, inl = cv2.solvePnPRansac(np.array(new_good_points), new_good_corners.points,
                                                       intrinsic_mat,
                                                       dist_coefs, rv, tv,
                                                       useExtrinsicGuess=1,
                                                       flags=meth,
                                                       reprojectionError=repr,
                                                       confidence=conf)
            while not succ:
                repr += 0.1
                conf -= 0.01
                if repr > 5:
                    break
                succ, rvec, tvec, inl = cv2.solvePnPRansac(np.array(new_good_points), new_good_corners.points,
                                                           intrinsic_mat,
                                                           dist_coefs, rv, tv,
                                                           useExtrinsicGuess=1,
                                                           flags=meth,
                                                           reprojectionError=repr,
                                                           confidence=conf)
            if not succ:
                continue
            rv = rvec
            tv = tvec
            view_mats[new_i] = _camtrack.rodrigues_and_translation_to_view_mat3x4(rv, tv)
            print('Recount of {} position, inliers count - {} from {}, cur repr - {}'
                  .format(new_i, len(inl), len(corner_storage[new_i].ids), repr))

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
