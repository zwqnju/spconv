# Copyright 2019 Yan Yan
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
from spconv import spconv_utils
from spconv.spconv_utils import (non_max_suppression, non_max_suppression_cpu,
                                 points_to_voxel_3d_np, points_to_voxel_3d_np_batch, rbbox_iou,
                                 rotate_non_max_suppression_cpu, rbbox_intersection)


def points_to_voxel_batch(points,
                     voxel_size,
                     coors_range_batch,
                     coor_to_voxelidx_batch,
                     max_points=35,
                     max_voxels=20000):
    """convert 3d points(N, >=3) to voxels. This version calculate
    everything in one loop. now it takes only 0.8ms(~6k voxels) 
    with c++ and 3.2ghz cpu.

    Args:
        points: [N, ndim] float tensor. points[:, :3] contain xyz points and
            points[:, 3:] contain other information such as reflectivity.
        voxel_size: [3] list/tuple or array, float. xyz, indicate voxel size
        coors_range: [6] list/tuple or array, float. indicate voxel range.
            format: xyzxyz, minmax
        coor_to_voxelidx: int array. used as a dense map.
        max_points: int. indicate maximum points contained in a voxel.
        max_voxels: int. indicate maximum voxels this function create.
            for voxelnet, 20000 is a good choice. you should shuffle points
            before call this function because max_voxels may drop some points.

    Returns:
        voxels: [M, max_points, ndim] float tensor. only contain points.
        coordinates: [M, 3] int32 tensor. zyx format.
        num_points_per_voxel: [M] int32 tensor.
    """
    if not isinstance(voxel_size, np.ndarray):
        voxel_size = np.array(voxel_size, dtype=points.dtype)
    if not isinstance(coors_range_batch, np.ndarray):
        coors_range_batch = np.array(coors_range_batch, dtype=points.dtype)
    voxelmap_shape = (coors_range_batch[0, 3:] - coors_range_batch[0, :3]) / voxel_size
    voxelmap_shape = tuple(np.round(voxelmap_shape).astype(np.int32).tolist())
    voxelmap_shape = voxelmap_shape[::-1]
    batch_size = coors_range_batch.shape[0]
    voxelmap_shape = (batch_size, ) + voxelmap_shape

    num_points_per_voxel_batch = np.zeros(shape=(batch_size, max_voxels), dtype=np.int32)
    voxels_batch = np.zeros(shape=(batch_size, max_voxels, max_points, points.shape[-1]), dtype=points.dtype)
    coors_batch = np.zeros(shape=(batch_size, max_voxels, 3), dtype=np.int32)
    voxel_num_batch = np.zeros(shape=(batch_size, ), dtype=np.int32)

    points_to_voxel_3d_np_batch(
        points, 
        voxels_batch, 
        coors_batch, 
        num_points_per_voxel_batch, 
        coor_to_voxelidx_batch,
        voxel_num_batch,
        coors_range_batch,
        voxel_size.tolist(), 
        max_points, 
        max_voxels,
    )
        
    return voxels_batch, coors_batch, num_points_per_voxel_batch, voxel_num_batch


class VoxelGenerator_Batch:
    def __init__(self,
                 voxel_size,            # [0.2, 0.2, 4] 保持不变，各分片大小一样
                 point_cloud_range_list,     # [-x, -y, -z, x, y, z] 改为list，每个元素一个
                 max_num_points,        # 30
                 max_voxels):           # 20000
        pc_range_batch = np.array(point_cloud_range_list, dtype=np.float32)
        voxel_size = np.array(voxel_size, dtype=np.float32)
        last_grid_size = None
        for point_cloud_range in pc_range_batch:
            grid_size = (
                point_cloud_range[3:] - point_cloud_range[:3]) / voxel_size
            if last_grid_size is not None:
                assert (last_grid_size == grid_size).all()
            last_grid_size = grid_size
        grid_size = np.round(grid_size).astype(np.int64)
        voxelmap_shape = tuple(np.round(grid_size).astype(np.int32).tolist())
        voxelmap_shape = voxelmap_shape[::-1]

        self._coor_to_voxelidx_batch = np.full(
            (pc_range_batch.shape[0],) + voxelmap_shape, 
            -1, dtype=np.int32
        )
        self._voxel_size = voxel_size
        self._point_cloud_range_batch = pc_range_batch
        self._max_num_points = max_num_points
        self._max_voxels = max_voxels
        self._grid_size = grid_size

    def generate(self, points, max_voxels=None):
        res = points_to_voxel_batch(
                    points, 
                    self._voxel_size, 
                    self._point_cloud_range_batch, 
                    self._coor_to_voxelidx_batch,
                    self._max_num_points, 
                    max_voxels or self._max_voxels
              )
        voxels_batch, coors_batch, num_points_per_voxel_batch, voxel_num_batch = res
        result = []
        for i, voxel_num in enumerate(voxel_num_batch):
            coors = coors_batch[i, :voxel_num]
            voxels = voxels_batch[i, :voxel_num]
            num_points_per_voxel = num_points_per_voxel_batch[i, :voxel_num]
            result.append((voxels, coors, num_points_per_voxel))

        return result

    def generate_multi_gpu(self, points, max_voxels=None):
        res = points_to_voxel(
            points, self._voxel_size, self._point_cloud_range, self._coor_to_voxelidx,
            self._max_num_points, max_voxels or self._max_voxels)
        return res


    @property
    def voxel_size(self):
        return self._voxel_size

    @property
    def max_num_points_per_voxel(self):
        return self._max_num_points


    @property
    def point_cloud_range(self):
        return self._point_cloud_range

    @property
    def grid_size(self):
        return self._grid_size



def points_to_voxel(points,
                     voxel_size,
                     coors_range,
                     coor_to_voxelidx,
                     max_points=35,
                     max_voxels=20000):
    """convert 3d points(N, >=3) to voxels. This version calculate
    everything in one loop. now it takes only 0.8ms(~6k voxels) 
    with c++ and 3.2ghz cpu.

    Args:
        points: [N, ndim] float tensor. points[:, :3] contain xyz points and
            points[:, 3:] contain other information such as reflectivity.
        voxel_size: [3] list/tuple or array, float. xyz, indicate voxel size
        coors_range: [6] list/tuple or array, float. indicate voxel range.
            format: xyzxyz, minmax
        coor_to_voxelidx: int array. used as a dense map.
        max_points: int. indicate maximum points contained in a voxel.
        max_voxels: int. indicate maximum voxels this function create.
            for voxelnet, 20000 is a good choice. you should shuffle points
            before call this function because max_voxels may drop some points.

    Returns:
        voxels: [M, max_points, ndim] float tensor. only contain points.
        coordinates: [M, 3] int32 tensor. zyx format.
        num_points_per_voxel: [M] int32 tensor.
    """
    if not isinstance(voxel_size, np.ndarray):
        voxel_size = np.array(voxel_size, dtype=points.dtype)
    if not isinstance(coors_range, np.ndarray):
        coors_range = np.array(coors_range, dtype=points.dtype)
    voxelmap_shape = (coors_range[3:] - coors_range[:3]) / voxel_size
    voxelmap_shape = tuple(np.round(voxelmap_shape).astype(np.int32).tolist())
    voxelmap_shape = voxelmap_shape[::-1]
    num_points_per_voxel = np.zeros(shape=(max_voxels, ), dtype=np.int32)
    voxels = np.zeros(
        shape=(max_voxels, max_points, points.shape[-1]), dtype=points.dtype)
    coors = np.zeros(shape=(max_voxels, 3), dtype=np.int32)
    voxel_num = points_to_voxel_3d_np(
        points, voxels, coors, num_points_per_voxel, coor_to_voxelidx,
        voxel_size.tolist(), coors_range.tolist(), max_points, max_voxels)
    # coors = coors[:voxel_num]
    # voxels = voxels[:voxel_num]
    # num_points_per_voxel = num_points_per_voxel[:voxel_num]
    return voxels, coors, num_points_per_voxel, voxel_num

class VoxelGenerator:
    def __init__(self,
                 voxel_size,
                 point_cloud_range,
                 max_num_points,
                 max_voxels=20000):
        point_cloud_range = np.array(point_cloud_range, dtype=np.float32)
        # [0, -40, -3, 70.4, 40, 1]
        voxel_size = np.array(voxel_size, dtype=np.float32)
        grid_size = (
            point_cloud_range[3:] - point_cloud_range[:3]) / voxel_size
        grid_size = np.round(grid_size).astype(np.int64)
        voxelmap_shape = tuple(np.round(grid_size).astype(np.int32).tolist())
        voxelmap_shape = voxelmap_shape[::-1]

        self._coor_to_voxelidx = np.full(voxelmap_shape, -1, dtype=np.int32)
        self._voxel_size = voxel_size
        self._point_cloud_range = point_cloud_range
        self._max_num_points = max_num_points
        self._max_voxels = max_voxels
        self._grid_size = grid_size

    def generate(self, points, max_voxels=None):
        res = points_to_voxel(
            points, self._voxel_size, self._point_cloud_range, self._coor_to_voxelidx,
            self._max_num_points, max_voxels or self._max_voxels)
        voxels, coors, num_points_per_voxel, voxel_num = res
        coors = coors[:voxel_num]
        voxels = voxels[:voxel_num]
        num_points_per_voxel = num_points_per_voxel[:voxel_num]

        return (voxels, coors, num_points_per_voxel)

    def generate_multi_gpu(self, points, max_voxels=None):
        res = points_to_voxel(
            points, self._voxel_size, self._point_cloud_range, self._coor_to_voxelidx,
            self._max_num_points, max_voxels or self._max_voxels)
        return res


    @property
    def voxel_size(self):
        return self._voxel_size

    @property
    def max_num_points_per_voxel(self):
        return self._max_num_points


    @property
    def point_cloud_range(self):
        return self._point_cloud_range

    @property
    def grid_size(self):
        return self._grid_size