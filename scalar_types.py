import warp as wp 
mat22 = wp.mat22d
mat33 = wp.mat33d
mat44 = wp.mat44d
vec2 = wp.vec2d
vec3 = wp.vec3d
vec4 = wp.vec4d
scalar = wp.float64
quat = wp.quatd
mat6 = wp.spatial_matrixd
vec6 = wp.spatial_vectord
vec12 = wp.types.vector(length=12, dtype=scalar)
mat12 = wp.types.matrix(shape = (12, 12), dtype = scalar)

mat34 = wp.types.matrix(shape = (3, 4), dtype = scalar)
mat24 = wp.types.matrix(shape = (2, 4), dtype = scalar)
mat12 = wp.types.matrix(shape = (12, 12), dtype = scalar)
mat99 = wp.types.matrix(shape = (9, 9), dtype = scalar)