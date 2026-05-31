"""Neural-network building blocks shared across detectors.

Both the TrackNet ball model (Step 5) and the TennisCourtDetector keypoint model (Step 4)
share the same VGG-style encoder/decoder, :class:`BallTrackerNet`, differing only in the
number of input/output channels. Keeping a single definition here means the public pretrained
``state_dict`` files from yastrebksv's repos load without modification.
"""

from .tracknet_arch import BallTrackerNet

__all__ = ["BallTrackerNet"]
