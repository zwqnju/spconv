// Copyright 2019 Yan Yan
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once
#include <pybind11/pybind11.h>
// must include pybind11/eigen.h if using eigen matrix as arguments.
// must include pybind11/stl.h if using containers in STL in arguments.
#include <algorithm>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
// #include <vector>
#include <iostream>
#include <math.h>

namespace spconv {
namespace py = pybind11;
using namespace pybind11::literals;



template <typename DType, int NDim>
void points_to_voxel_3d_np_batch(
            py::array_t<DType> points, 
            py::array_t<DType> voxels_batch,
            py::array_t<int> coors_batch,
            py::array_t<int> num_points_per_voxel_batch,
            py::array_t<int> coor_to_voxelidx_batch,
            py::array_t<int> voxel_num_batch,
            py::array_t<DType> coors_range_batch,
            std::vector<DType> voxel_size,
            int max_points,
            int max_voxels) {
  auto points_rw = points.template mutable_unchecked<2>();
  auto voxels_batch_rw = voxels_batch.template mutable_unchecked<4>();
  auto coors_batch_rw = coors_batch.mutable_unchecked<3>();
  auto num_points_per_voxel_batch_rw = num_points_per_voxel_batch.mutable_unchecked<2>();
  auto coor_to_voxelidx_batch_rw = coor_to_voxelidx_batch.mutable_unchecked<NDim+1>();
  auto voxel_num_batch_rw = voxel_num_batch.mutable_unchecked<1>();
  auto coors_range_batch_rw = coors_range_batch.template mutable_unchecked<2>();

  auto N = points_rw.shape(0);
  auto num_features = points_rw.shape(1);
  constexpr int ndim_minus_1 = NDim - 1;
  
  int batch_size = voxel_num_batch_rw.shape(0);
  bool failed = false;
  int coor[NDim];         // 某个点所在的voxel坐标：x, y, z
  int c;

  int grid_size[NDim];      // voxel尺寸：0.2, 0.2, 4
  for (int i = 0; i < NDim; ++i) {
    grid_size[i] = round((
          coors_range_batch_rw(0, NDim + i) - coors_range_batch_rw(0, i)) / voxel_size[i]
    );
  }
  int j, b, k;
  int voxelidx, num;        // voxel编号
  for (int i = 0; i < N; ++i) { // 遍历每个点
    for (b = 0; b < batch_size; ++b) {
      failed = false;
      for (j = 0; j < NDim; ++j) {
        c = floor((points_rw(i, j) - coors_range_batch_rw(b, j)) / voxel_size[j]);
        if ((c < 0 || c >= grid_size[j])) {
          failed = true;
          break;
        }
        coor[ndim_minus_1 - j] = c;
      }
      if (failed)
        continue;
      
      voxelidx = coor_to_voxelidx_batch_rw(b, coor[0], coor[1], coor[2]);
      if (voxelidx == -1) {
        voxelidx = voxel_num_batch_rw(b);
        if (voxelidx >= max_voxels)
          continue;
        voxel_num_batch_rw(b) += 1;
        coor_to_voxelidx_batch_rw(b, coor[0], coor[1], coor[2]) = voxelidx;
        for (k = 0; k < NDim; ++k) {
          coors_batch_rw(b, voxelidx, k) = coor[k];
        }
      }
      num = num_points_per_voxel_batch_rw(b, voxelidx);
      if (num < max_points) {
        for (k = 0; k < num_features; ++k) {
          voxels_batch_rw(b, voxelidx, num, k) = points_rw(i, k);
        }
        num_points_per_voxel_batch_rw(b, voxelidx) += 1;
      }
    }
  }
  int voxel_num;
  for (b = 0; b < batch_size; ++b) {
    voxel_num = voxel_num_batch_rw(b);
    for (int i = 0; i < voxel_num; ++i) {
      coor_to_voxelidx_batch_rw(b, coors_batch_rw(b, i, 0), coors_batch_rw(b, i, 1), coors_batch_rw(b, i, 2)) = -1;
    }
  }
}



template <typename DType, int NDim>
int points_to_voxel_3d_np(py::array_t<DType> points, py::array_t<DType> voxels,
                          py::array_t<int> coors,
                          py::array_t<int> num_points_per_voxel,
                          py::array_t<int> coor_to_voxelidx,
                          std::vector<DType> voxel_size,
                          std::vector<DType> coors_range, int max_points,
                          int max_voxels) {
  auto points_rw = points.template mutable_unchecked<2>();
  auto voxels_rw = voxels.template mutable_unchecked<3>();
  auto coors_rw = coors.mutable_unchecked<2>();
  auto num_points_per_voxel_rw = num_points_per_voxel.mutable_unchecked<1>();
  auto coor_to_voxelidx_rw = coor_to_voxelidx.mutable_unchecked<NDim>();
  auto N = points_rw.shape(0);
  auto num_features = points_rw.shape(1);
  // auto ndim = points_rw.shape(1) - 1;
  constexpr int ndim_minus_1 = NDim - 1;
  int voxel_num = 0;
  bool failed = false;
  int coor[NDim];
  int c;
  int grid_size[NDim];
  for (int i = 0; i < NDim; ++i) {
    grid_size[i] =
        round((coors_range[NDim + i] - coors_range[i]) / voxel_size[i]);
  }
  int voxelidx, num;
  for (int i = 0; i < N; ++i) {
    failed = false;
    for (int j = 0; j < NDim; ++j) {
      c = floor((points_rw(i, j) - coors_range[j]) / voxel_size[j]);
      if ((c < 0 || c >= grid_size[j])) {
        failed = true;
        break;
      }
      coor[ndim_minus_1 - j] = c;
    }
    if (failed)
      continue;
    voxelidx = coor_to_voxelidx_rw(coor[0], coor[1], coor[2]);
    if (voxelidx == -1) {
      voxelidx = voxel_num;
      if (voxel_num >= max_voxels)
        break;
      voxel_num += 1;
      coor_to_voxelidx_rw(coor[0], coor[1], coor[2]) = voxelidx;
      for (int k = 0; k < NDim; ++k) {
        coors_rw(voxelidx, k) = coor[k];
      }
    }
    num = num_points_per_voxel_rw(voxelidx);
    if (num < max_points) {
      for (int k = 0; k < num_features; ++k) {
        voxels_rw(voxelidx, num, k) = points_rw(i, k);
      }
      num_points_per_voxel_rw(voxelidx) += 1;
    }
  }
  for (int i = 0; i < voxel_num; ++i) {
    coor_to_voxelidx_rw(coors_rw(i, 0), coors_rw(i, 1), coors_rw(i, 2)) = -1;
  }
  return voxel_num;
}

} // namespace spconv