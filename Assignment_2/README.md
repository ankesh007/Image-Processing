# Face Morphing, Swapping and Snapchat Filtering

## Problem Statement

 Part A: Face Morphing Implement triangle based approach for image morphing for face images. This would require:

    Establishing correspondence of features, to beign with these may be provided interactively.
    Delaunay triangulation or any other triangulation available in the libray may be used for the purpose of obtaining the triangulation of feature points. One may have to add feature points at the boundary of the image to include the entire image.
    Creating intermediate frames using Barycentric coordinates. 

Part B: Face Swappingg The face morphing may be adapted for swapping two faces. This may require the following:

    Consider only the portion of the face that would be swapped. That is the tirnagulation will limit to the region to swap.
    Color adjustment to the respective sources.
    The boundary of the region would need smoothing to avoid sharp edges.
    Pose/alignment adjustment. 

Part C: Filters for augmenting face This entails augmenting face image with filters (as used in snapchat). This may require the following.

    Consider a set of filters like masks aligned on a template.
    Placement of the filter on the desired location
    Compositing with the original image. This may require defining appropriate alpha-map for the mask. 


## Image Morphing(Part A)

To run image morphing algorithm:  
`python morphing.py --source <SOURCE_IMAGE_PATH> --dest <DESTINATION_IMAGE_PATH> [--k no_of_interim_images] --output OUTPUT_IMAGE_PATH`

Note:  
1. Default no. of interim images=10
2. ith Images generated is saved as: OUTPUT_IMAGE_PATH_i.jpg 

### Assumptions

1. Its assumed that topograph of triangles are same in point correspondences selected by user in source and destination images.
