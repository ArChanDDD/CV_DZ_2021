#! /usr/bin/env python3

__all__ = [
    'FrameCorners',
    'CornerStorage',
    'build',
    'dump',
    'load',
    'draw',
    'calc_track_interval_mappings',
    'calc_track_len_array_mapping',
    'without_short_tracks'
]

import click
import cv2
import numpy as np
import pims

from _corners import (
    FrameCorners,
    CornerStorage,
    StorageImpl,
    dump,
    load,
    draw,
    calc_track_interval_mappings,
    calc_track_len_array_mapping,
    without_short_tracks,
    create_cli
)


class _CornerStorageBuilder:

    def __init__(self, progress_indicator=None):
        self._progress_indicator = progress_indicator
        self._corners = dict()

    def set_corners_at_frame(self, frame, corners):
        self._corners[frame] = corners
        if self._progress_indicator is not None:
            self._progress_indicator.update(1)

    def build_corner_storage(self):
        return StorageImpl(item[1] for item in sorted(self._corners.items()))


def _build_impl(frame_sequence: pims.FramesSequence,
                builder: _CornerStorageBuilder) -> None:
    def make_mask(img, corners, min_d):
        my_mask = np.uint8(np.zeros_like(img) + 255)
        for j in corners:
            # print('j= ', j, 'sh = ', img[i].shape)
            ss = my_mask[max(j[1] - min_d, 0):min(j[1] + min_d, img.shape[0]),
                 max(j[0] - min_d, 0):min(j[0] + min_d, img.shape[1])].shape
            my_mask[max(j[1] - min_d, 0):min(j[1] + min_d, img.shape[0]),
            max(j[0] - min_d, 0):min(j[0] + min_d, img.shape[1])] = np.zeros(ss)
        return my_mask

    # TODO
    ncorners = 5000
    image_0 = frame_sequence[0]
    img = [0, 0, 0, 0, 0]
    img[0] = cv2.resize(image_0, (0, 0), fx=0.5, fy=0.5)
    img[1] = cv2.resize(img[0], (0, 0), fx=0.5, fy=0.5)
    img[2] = cv2.resize(img[1], (0, 0), fx=0.5, fy=0.5)
    img[3] = cv2.resize(img[2], (0, 0), fx=0.5, fy=0.5)
    img[4] = cv2.resize(img[3], (0, 0), fx=0.5, fy=0.5)
    # print(img[4].shape, img[3].shape, img[2].shape)
    corners0 = np.array([])
    corners_last = np.array([])
    for i in range(4, -1, -1):
        my_mask = make_mask(img[i], corners_last.reshape(-1, 2).astype(int), 8)
        corners = cv2.goodFeaturesToTrack(img[i], maxCorners=ncorners, mask=my_mask, qualityLevel=0.01, minDistance=8,
                                          blockSize=5)
        # print(corners)
        if len(corners_last) == 0:
            corners_last = corners.reshape(-1, 2) * 2
        else:
            corners_last = np.vstack([corners_last * 2, corners.reshape(-1, 2) * 2])
        corners = corners.reshape(-1, 2)
        corners = corners * (2 ** (i + 1))
        if i == 4:
            corners0 = corners
        else:
            corners0 = np.vstack([corners0, corners])

    my_mask = make_mask(image_0, corners_last.reshape(-1, 2).astype(int), 8)
    corners1 = cv2.goodFeaturesToTrack(image_0, maxCorners=ncorners, mask=my_mask, qualityLevel=0.01, minDistance=8,
                                       blockSize=5)
    num_corners = corners1.shape[0]
    corners1 = corners1.reshape(num_corners, 2)
    corners0 = np.vstack([corners0, corners1])
    ncorners = corners0.shape[0]
    sizes = np.zeros(ncorners)
    nums = np.array(np.arange(ncorners))
    sizes.fill(10)
    corners = FrameCorners(
        nums,
        np.array(corners0),
        sizes
    )
    builder.set_corners_at_frame(0, corners)
    for frame, image_1 in enumerate(frame_sequence[1:], 1):
        corners1, _st, _err = cv2.calcOpticalFlowPyrLK(prevImg=np.uint8(image_0 * 255),
                                                       nextImg=np.uint8(image_1 * 255),
                                                       prevPts=corners0,
                                                       nextPts=None,
                                                       maxLevel=5,
                                                       winSize=(7, 7),
                                                       criteria=(
                                                           cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
        corners1 = corners1[np.where(_st == 1)[0]]
        # print(len(np.where(_st == 0)[0]))
        nums = nums[np.where(_st == 1)[0]]
        sizes = sizes[np.where(_st == 1)[0]]
        corners1 = corners1.reshape(-1, 2)
        max_num = max(nums)
        my_mask = make_mask(image_1, corners1.astype(int), 10)
        new_corners = cv2.goodFeaturesToTrack(image_1, maxCorners=int(corners1.shape[0] * 0.025), corners=corners1, mask=my_mask,
                                              qualityLevel=0.01,
                                              minDistance=10, blockSize=5)
        if new_corners is not None:
            new_corners = new_corners.reshape(-1, 2)
            corners1 = np.vstack([corners1, new_corners])
            nums = np.hstack([nums, np.arange(max_num + 1, max_num + 1 + new_corners.shape[0], 1)])
            sizes = np.hstack([sizes, np.ones(new_corners.shape[0]) + 10])
        corners = FrameCorners(
            nums,
            np.array(corners1),
            sizes
        )
        builder.set_corners_at_frame(frame, corners)
        image_0 = image_1
        corners0 = corners1


def build(frame_sequence: pims.FramesSequence,
          progress: bool = True) -> CornerStorage:
    """
    Build corners for all frames of a frame sequence.

    :param frame_sequence: grayscale float32 frame sequence.
    :param progress: enable/disable building progress bar.
    :return: corners for all frames of given sequence.
    """
    if progress:
        with click.progressbar(length=len(frame_sequence),
                               label='Calculating corners') as progress_bar:
            builder = _CornerStorageBuilder(progress_bar)
            _build_impl(frame_sequence, builder)
    else:
        builder = _CornerStorageBuilder()
        _build_impl(frame_sequence, builder)
    return builder.build_corner_storage()


if __name__ == '__main__':
    create_cli(build)()  # pylint:disable=no-value-for-parameter
