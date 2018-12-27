# Image-Analogies, Inpainting, Laplacian Pyramids

## Problem Statement

  Implement image pyramid (Gaussian and Laplacian) as discussed in the class - may refer to paper. You are required to apply this for image compression and mosacing. Implement image inpainting. Implement image analogies.

Note: Arguments in `<>` brackets are mandatory whereas arguments asked in `[]` have default values hard-coded

## Mosaicing

To run image mosaicing algorithm:  
`python3 mosaicing.py --source <SOURCE_IMAGE_PATH> --dest <SOURCE_IMAGE2_PATH> [--levels no_of_levels_of_pyramid] --output OUTPUT_IMAGE_FOLDER`

## Compression

To run image compression algorithm:  
`python3 compression_median_cut_binning.py --source <SOURCE_IMAGE_PATH> [--levels no_of_levels_of_pyramid] [--to_keep no._of_bins/colors retained] [--mode median_cut/binning] --output OUTPUT_IMAGE_FOLDER`

## Image Analogies

To run image analogy algorithm:  
`python3 image_analogies.py --source <SOURCE_IMAGE_PATH> --source_filtered <FILTERED_SOURCE_IMAGE_PATH> --dest <DEST_UNFILTERED_PATH> [--levels no_of_levels_of_pyramid] [--mode RGB/YIQ] --output OUTPUT_IMAGE_FOLDER [--resize_height resize_height_to_this] [--kappa hyperparameter]`

Notes:

1. `YIQ/RGB` - Use `RGB` mode only if task is of transferring color details.
2. `resize_height` - Kept low as task is computationally very expensive
3. `kappa` - hyperparameter -> not to be changed if beginner

## Inpainting

To run image inpainting algorithm:

`python3 inpaint.py`

Note: The file works with hard_coded image path. Please make necessary changes to make it an argument. I am lazy to make this change. 
