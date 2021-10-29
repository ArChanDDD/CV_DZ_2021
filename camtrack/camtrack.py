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
    # REMOVE BEFORE FLIGHT #
    # room - 1
    # fox_head_short_0_20 - 0
    # bike_t_0_10 - 2
    # 0_30 - 3
    # fox_0_45 - 4
    # house 0_10 - 5
    # iron_man - 6
    known_views = \
        [(0, Pose(np.array([[0.9338749319308997, 0.10174056676185345, 0.3428213362467898],
                            [-0.18410948455155535, 0.9586460660229594, 0.21702861054907982],
                            [-0.30656371150091827, -0.2657942384132515, 0.9139869329575239]]),
                  np.array([-0.8495888218369549, -0.4800792734292757, -2.4116739864531413]))),
         (20, Pose(np.array([[0.9173107208741516, 0.09354338611117231, 0.3870280045218957],
                             [-0.18512260364832947, 0.9607628899986013, 0.20655336070855668],
                             [-0.35252044336152744, -0.2611212440924555, 0.8986350944047313]]),
                   np.array([-0.946330637081571, -0.46502329879466053, -2.4156466648751653])))], \
        [(0, Pose(np.array([[0.8638654850974964, -0.12589307069435537, -0.4877369766671514],
                            [0.19548778466300357, 0.9761640157340209, 0.09427799548983523],
                            [0.464242339414253, -0.1767901273747386, 0.8678849584812525]]),
                  np.array([-1.4614250000023143, -1.4521270000009687, -1.5202050000013467]))),
         (20, Pose(np.array([[0.9475194471167725, -0.08189403864427115, -0.30903116957688337],
                             [0.11626198381202348, 0.9887167580071022, 0.09445805183264291],
                             [0.2978087447601541, -0.12542941788295156, 0.9463495192971386]]),
                   np.array([-1.4156090000006023, -1.4590129999989312, -1.497897999999159])))], \
        [(0, Pose(np.array([[0.7043087872871573, 0.4082858766909472, -0.5807338246088294],
                            [-0.7098499442785533, 0.41413630804335666, -0.569740445262536],
                            [0.007885984899093701, 0.8135070751105503, 0.5815015477083862]]),
                  np.array([0.3116233108186326, 0.30418567442747546, -0.41096351716002216]))),
         (10, Pose(np.array([[0.7043186331539746, 0.40868978713295806, -0.5804376976776529],
                             [-0.7098423900765792, 0.4143125246972549, -0.569621728105026],
                             [0.007684025148739504, 0.8132144795637889, 0.5819133663916902]]),
                   np.array([0.304993829088243, 0.311569357802947, -0.40951183077214115])))], \
        [(0, Pose(np.array([[0.7043087872871573, 0.4082858766909472, -0.5807338246088294],
                            [-0.7098499442785533, 0.41413630804335666, -0.569740445262536],
                            [0.007885984899093701, 0.8135070751105503, 0.5815015477083862]]),
                  np.array([0.3116233108186326, 0.30418567442747546, -0.41096351716002216]))),
         (30, Pose(np.array([[0.7048034025101665, 0.4080742330494248, -0.5802823313967228],
                             [-0.7093577703711065, 0.41460702521180437, -0.5700110246821115],
                             [0.007982319476496247, 0.8133734904495855, 0.5816870701757374]]),
                   np.array([0.326927677970979, 0.326498359702827, -0.3572384378337068])))], \
        [(0, Pose(np.array([[0.9338749319308997, 0.10174056676185345, 0.3428213362467898],
                            [-0.18410948455155535, 0.9586460660229594, 0.21702861054907982],
                            [-0.30656371150091827, -0.2657942384132515, 0.9139869329575239]]),
                  np.array([-0.8495888218369549, -0.4800792734292757, -2.4116739864531413]))),
         (45, Pose(np.array([[0.8894728595096433, 0.08152118441526202, 0.4496580130357639],
                             [-0.18568030697752663, 0.9635479925690426, 0.1926086436713052],
                             [-0.41756539104249274, -0.2548127989479751, 0.8721866667725603]]),
                   np.array([-1.0745375654597404, -0.4435578409462959, -2.411496757949314])))], \
        [(0, Pose(np.array([[-0.5629978404903484, 0.4150358824112204, -0.7146877975132525],
                            [-0.8263772439031485, -0.29481795030308744, 0.4797739331582349],
                            [-0.011579393863255183, 0.8607134206518601, 0.5089580779863246]]),
                  np.array([0.5863733141675634, -0.4511346431620138, -0.3452258899955657]))),
         (10, Pose(np.array([[-0.604352147465568, 0.4010300978537621, -0.688428167980624],
                             [-0.7965479533506925, -0.32195354744433313, 0.5117199148956707],
                             [-0.016426803297635573, 0.8576250777020291, 0.5140130214595826]]),
                   np.array([0.5839539098622751, -0.40746340558657573, -0.345694628346105])))], \
        [(0, Pose(np.array([[-0.694752223762706, 0.4072531311662769, -0.5928441909406199],
                            [-0.7192399217483174, -0.3975540280964003, 0.5697760346906502],
                            [-0.003644521950722346, 0.8222503766491019, 0.5691142552775855]]),
                  np.array([0.31668383640312137, -0.30701112077061077, -0.40365927596201046]))),
         (10, Pose(np.array([[-0.6928179727902676, 0.4067199840264796, -0.5954679766135911],
                             [-0.7211014064304828, -0.395338455083003, 0.5689642058834319],
                             [-0.004002277197270865, 0.8235814231307581, 0.5671839395215202]]),
                   np.array([0.33790987798492766, -0.31958343057578725, -0.3532927185077867])))]

    #known_view_1, known_view_2 = known_views[6]
    # END OF REMOVE #

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
                                                   flags=cv2.SOLVEPNP_ITERATIVE,
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
