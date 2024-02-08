## hw2_walkthrough1
Threshold was chosen by looking at histogram of image values.

## hw2_challenge1

### compute2DProperties

I did not add any additional properties.

### recognizeObjects

The criteria I used was roundness. My Threshold was +- 0.05

If this hadn't worked my plan was to segment image for each object and rotate the image to its orientation. I would then calclulate the min/max y and x cordinates of the object and draw a box around the object. Within the box I would calculate the pixel ratio of black to white pixels.