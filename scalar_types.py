import warp as wp 

mat33 = wp.mat33d
mat44 = wp.mat44d
vec4 = wp.vec4d
vec3 = wp.vec3d
scalar = wp.float64
quat = wp.quatd
mat6 = wp.spatial_matrixd
vec6 = wp.spatial_vectord

vec12 = wp.types.vector(length=12, dtype=scalar)
mat12 = wp.types.matrix(shape = (12, 12), dtype = scalar)