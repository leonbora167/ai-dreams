"""Stateless Inference Services subsystem."""
from .detector_service import (
    BaseDetectorService,
    Detection,
    LocalRFDETRService,
    LocalYOLOService,
    TritonDetectorService,
    create_detector_service
)
from .feature_service import (
    BaseReIDExtractor,
    LocalOSNetExtractor,
    TritonReIDExtractor,
    StatelessColorExtractor,
    StatelessPoseExtractor,
    StatelessFeatureService
)
from .speed_service import (
    BaseSpeedEstimator,
    HeightPriorSpeedEstimator,
    BiomechanicalSpeedEstimator,
    AutoHomographySpeedEstimator,
    MonocularDepthSpeedEstimator,
    TrackSpeedTracker,
    create_speed_tracker
)

__all__ = [
    'BaseDetectorService',
    'Detection',
    'LocalRFDETRService',
    'LocalYOLOService',
    'TritonDetectorService',
    'create_detector_service',
    'BaseReIDExtractor',
    'LocalOSNetExtractor',
    'TritonReIDExtractor',
    'StatelessColorExtractor',
    'StatelessPoseExtractor',
    'StatelessFeatureService',
    'BaseSpeedEstimator',
    'HeightPriorSpeedEstimator',
    'BiomechanicalSpeedEstimator',
    'AutoHomographySpeedEstimator',
    'MonocularDepthSpeedEstimator',
    'TrackSpeedTracker',
    'create_speed_tracker'
]

