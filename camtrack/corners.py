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
    # TODO
    ncorners = 5000
    image_0 = frame_sequence[0]
    corners0 = cv2.goodFeaturesToTrack(image_0, maxCorners=ncorners, qualityLevel=0.01, minDistance=3, blockSize=5)
    ncorners = corners0.shape[0]
    corners0 = corners0.reshape(ncorners, 2)
    sizes = np.zeros(ncorners)
    sizes.fill(10)
    corners = FrameCorners(
        np.array(np.arange(ncorners)),
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
                                                       winSize=(7, 7))
        corners1 = corners1.reshape(corners1.shape[0], 2)
        corners = FrameCorners(
            np.array(np.arange(corners1.shape[0])),
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
