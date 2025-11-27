import numpy as np

from app.voxel.Voxel import Voxel



v1 = Voxel([1, 2, 3], 3)
v2 = Voxel([1, 20, 3], 3)
v3 = Voxel([1, 2, 30], 3)
v4 = Voxel([1, 2, 3], 3)

print(v1, hash(v1))
print(v2, hash(v2))
print(v3, hash(v3))
print(v4, hash(v4))

print(v1 == v4)
print(v1 == v2)

set_ = {v1, v2, v3, v4}

print(set_)