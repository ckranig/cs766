# Challenge 1b

I solved the equation: ρ = x cos θ + y sin θ
For each theta bin I chose the rho bin that was closest to x cos θ + y sin θ. I only chose a single bin as opposed to a patch of bins.
This single bin approach gave me good results so I kept it.

# Challenge 1c

I used a simple threshold method where I chose any entries in the accumulator that were above the threshold to draw lines for.

# Challenge 1d

For each line chosen I did a matrix multiplication with the original edge image and drew red pixels for each x,y where edge_img[x,y] >0 and line_img[x,y] > 0.