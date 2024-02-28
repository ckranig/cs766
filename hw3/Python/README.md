# Challenge 1b

I solved the equation: ρ = x cos θ + y sin θ
For each theta bin I chose the rho bin that was closest to x cos θ + y sin θ. I only chose a single bin as opposed to a patch of bins.
This single bin approach gave me good results so I kept it.
For the number of bins I had one θ bin per degree between -90 and 90, and I had one bin for each integer between -Image_Diagonal and Image_Diagonal.

# Challenge 1c

I started off by sorting all of the lines by their accumulator value. I then grouped all lines that where within 5 indices of each other before using a simple threshold method to weed out unwanted lines.

# Challenge 1d

For each line chosen I did a matrix multiplication with the original edge image and drew red pixels for each x,y where edge_img[x,y] >0 and line_img[x,y] > 0.