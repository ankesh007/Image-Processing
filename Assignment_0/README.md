# Image Scaling

## Problem Statement

This is a warm up assignment as discussed in the class. The basic intent of this assignment is to perform the scaling operation to an image. The scaling up should be done a) replication b) interpolation. The scaling down should be done with the corresponding operations as would be used for scaling up. Consider a window (a rectangular area) of a given size for the display within which the zoomed-in or out area is shown when placed on the image. 

## Scaling

To run code that scales entire image:

`python scaling.py [-h] --input INPUT_IMAGE_PATH --output OUTPUT_IMAGE_PATH [--scale SCALE] [--interpolate] [--replicate]`

Note:  
`--input` and `--output` are mandatory flags  
`--scale` defaults to 1  

If none of the scaling methods flags are set, the mode defaults to `replication`  
If both flags are set, mode defaults to `interpolation`  


### Results

Some results are displayed. To interpret, check name of image. The name is in format:
``
OriginalName?{_R}{SCALE}.jpg
``
_\_R_ represents Replication mode was used to generate
dot represents . for decimal