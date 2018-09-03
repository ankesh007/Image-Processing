# Painterly Rendered Images

## Problem Statement

This assignment deals with processing of images so that these can be converted into painterly rendered images. There are two parts to the assignment. The first part of the assignment requires implementation of parts from the paper on color quantization(***color_quantization_refpaper.pdf***). These are:
```
1. Polpularity algorithm
2. Median cut algorithm
3. Floyed and Steinberg algorithm 
```
The second part of the assignment is implementation of edge detection using eXtended Difference of Gaussian (xDoG) from the paper (***ref2_xdog.pdf***).

A final result in terms of non-photorealistic (or painterly) rendered output you are required to combine the color quantization from fist part with edges obtained from the second part. 


## Colour Quantization

Check out `color_quantization` folder for different quantization algorithms.

To run code, change to folder: `cd color_quantization`

To run popularity algorithm:  
`python popularity.py <input_image_path> <output_image_path> <color_palette> <dither(True/False)>`

To run median_cut algorithm:  
`python median_cut.py <input_image_path> <output_image_path> <color_palette> <dither(True/False)>`

## XDOG 

To run code

`python xdog.py <input_image_path> <output_image_path>`


## Painted render 

To run code

`python painter.py <input_image_path> <output_image_path>`


### Results

Some results are displayed. To interpret, check name of image. The name is in format:
``
OriginalName_{P|M}{D}{COLOR_PALETTE}.jpg
``
M|P indicates Median-cut|Popularity
D   indicates Dithering

### Assumptions

1. Its assumed that pixel-level of each color map is in range [0,255]. To change upper-bound, open code and edit `color_levels` at top.
