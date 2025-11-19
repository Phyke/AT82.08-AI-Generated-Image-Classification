"""Type definitions for static type checking."""

import numpy as np
from numpy.typing import NDArray

type RGBImage_uint8 = NDArray[np.uint8]
type GrayImage_uint8 = NDArray[np.uint8]
type GrayImage_float32 = NDArray[np.float32]
type ColorImage_uint8 = NDArray[np.uint8]
type ColorImage_float32 = NDArray[np.float32]
type FeatureVector_float32 = NDArray[np.float32]
type FeatureMatrix_float32 = NDArray[np.float32]
type FeatureMatrix_float64 = NDArray[np.float64]
