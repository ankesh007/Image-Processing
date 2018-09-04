# Painterly Rendered Images

## Problem Statement

This is a warm up assignment as discussed in the class. The basic intent of this assignment is to perform the scaling operation to an image. The scaling up should be done a) replication b) interpolation. The scaling down should be done with the corresponding operations as would be used for scaling up. Consider a window (a rectangular area) of a given size for the display within which the zoomed-in or out area is shown when placed on the image. 

## Scaling

To run code that scales entire image:

`python scaling.py <input_image_path> <output_image_path> <scale_factor:1 for no scaling>`


### Results

Some results are displayed. To interpret, check name of image. The name is in format:
``
OriginalName?{_dot}{SCALE_FACTOR_WITHOUT_DECIMALS}.jpg
``
The dot part indicates downscaling - _dot_25_ means scale factor is 0.25
