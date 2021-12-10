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
    frame_count = len(corner_storage)

    def calc_views():
        print('starting counting first frames...')
        view_pairs = []
        for fst in range(frame_count):
            for snd in range(fst + 15, min(fst + 60, frame_count)):
                view_pairs.append([fst, snd])

        seed = 307
        np.random.seed(seed)
        np.random.shuffle(view_pairs)
        max_len = 0
        good_id_1, good_id_2 = view_pairs[0]
        good_view_mat = _camtrack.eye3x4()
        max_corners_len = max([len(x.ids) for x in corner_storage])
        min_commons = max_corners_len
        prob = 0.9
        while max_len == 0 or min_commons > max_corners_len * 0.5:
            print('Cur commons - {}'.format(min_commons))
            counter = 0
            if min_commons < 100:
                min_commons = max_corners_len
                prob -= 0.1
            for [fst, snd] in view_pairs:
                counter += 1
                if counter >= 500:
                    break
                frame1 = corner_storage[fst]
                frame2 = corner_storage[snd]
                if len([x for x in frame1.ids if x in frame2.ids]) < min_commons:
                    continue
                corr_matcher = _camtrack.build_correspondences(frame1, frame2)
                essential_mat, mask = cv2.findEssentialMat(corr_matcher.points_1,
                                                           corr_matcher.points_2,
                                                           intrinsic_mat,
                                                           method=cv2.RANSAC,
                                                           prob=prob)
                if essential_mat is None:
                    continue

                new_mask = mask.flatten().astype(bool)
                common_ids = corr_matcher.ids[new_mask]
                common_points_1 = corr_matcher.points_1[new_mask]
                common_points_2 = corr_matcher.points_2[new_mask]
                correspondence = _camtrack.Correspondences(common_ids, common_points_1, common_points_2)

                homography_mat, mask = cv2.findHomography(correspondence.points_1,
                                                          correspondence.points_2,
                                                          method=cv2.RANSAC,
                                                          ransacReprojThreshold=1,
                                                          confidence=0.999)
                if homography_mat is None or np.count_nonzero(mask) / len(correspondence.ids) > 0.2:
                    continue

                pos_rvec1, pos_rvec2, pos_tvec = cv2.decomposeEssentialMat(essential_mat)
                pos_views = [np.hstack([pos_rvec1, pos_tvec]),
                             np.hstack([pos_rvec2, pos_tvec]),
                             np.hstack([pos_rvec1, -pos_tvec]),
                             np.hstack([pos_rvec2, -pos_tvec])]
                for view_mat in pos_views:
                    points_3d, correspondence_ids, mcos = _camtrack.triangulate_correspondences(correspondence,
                                                                                                view_mat,
                                                                                                _camtrack.eye3x4(),
                                                                                                intrinsic_mat,
                                                                                                _camtrack.TriangulationParameters(
                                                                                                    1, 1, 0.00001))
                    if len(points_3d) > max_len:
                        max_len = len(points_3d)
                        good_view_mat = view_mat
                        good_id_1 = fst
                        good_id_2 = snd
                        print('new max_len of common points after triangulation: {}. Frames ids: {} and {}'.format(
                            max_len,
                            fst,
                            snd))
            min_commons *= 0.9

        print('Done!')
        return [good_id_1, _camtrack.view_mat3x4_to_pose(good_view_mat)], \
               [good_id_2, _camtrack.view_mat3x4_to_pose(_camtrack.eye3x4())],

    if known_view_1 is None or known_view_2 is None:
        known_view_1, known_view_2 = calc_views()
    triangulation_parameters = _camtrack.TriangulationParameters(1, 10, 0.00001)
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
        success, rv, tv, inliers = cv2.solvePnPRansac(np.array(good_points), good_corners.points, intrinsic_mat,
                                                      dist_coefs, rvec, tvec,
                                                      useExtrinsicGuess=1,
                                                      flags=cv2.SOLVEPNP_ITERATIVE,
                                                      reprojectionError=1,
                                                      confidence=0.99999)
        next_rvecs[next_i_counter % 4] = rv
        next_tvecs[next_i_counter % 4] = tv
        conf = 0.9999
        repr = 1
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
        # if success:
        #    if len(inliers) / len(cur_corners.ids) < 0.03:
        #        print('skipped ', i, ' and ', len(inliers) / len(cur_corners.ids))
        #        next_i_counter += 1
        #        continue
        view_mat = _camtrack.rodrigues_and_translation_to_view_mat3x4(rv, tv)
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
            succ, rvec, tvec, inl = cv2.solvePnPRansac(np.array(new_good_points), new_good_corners.points,
                                                       intrinsic_mat,
                                                       dist_coefs, rv, tv,
                                                       useExtrinsicGuess=1,
                                                       flags=meth,
                                                       reprojectionError=2,
                                                       confidence=0.9999999)
            conf = 0.999999
            repr = 1

            while not succ:
                repr += 0.1
                conf -= 0.00001
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
