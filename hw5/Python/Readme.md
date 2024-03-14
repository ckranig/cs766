## Challenge 1a
Results 

Sphere center: (242.8846852816852, 250.36586443366767)

Sphere radius: 178.0525476769344

## Challenge 1b

We assume that the point given is on the sphere.

Formula:

x' = cord_x - center_x

y' = cord_x - center_y

z' = sqrt(radius^2 - x_prime^2 - y_prime^2)

n = (x',y',z')

Notice that z is normally negative since we have a left hand cordinate system and the image is from the front of the shpere.

We can assume that the brightest pixel on the sphere is the direction of the light source since we began by assuming that the spherical object is Lambertian and uniform. This means that eradiance is maximized when the normal vector of a point on the sphere aligns with the vector directed towards the light source (See below formula).

E = (J/r^2) * (n*s)