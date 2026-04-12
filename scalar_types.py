import warp as wp 
mat22 = wp.mat22d
mat33 = wp.mat33d
mat44 = wp.mat44d
vec4 = wp.vec4d
vec3 = wp.vec3d
vec2 = wp.vec2d
scalar = wp.float64
quat = wp.quatd
mat6 = wp.spatial_matrixd
vec6 = wp.spatial_vectord

mat34 = wp.types.matrix(shape = (3, 4), dtype = scalar)
mat24 = wp.types.matrix(shape = (2, 4), dtype = scalar)
