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
#include <vector>
#include <iostream>
#include <math.h>

std::vector<int> init_vector(int bit_count) {
  struct Number {
    std::vector<int> bit_list;
    Number(int N): bit_list(N) {
      for (int i = 0; i < N; ++i) {
        bit_list[i] = 0;
      }
    }
    int get_number() {
      int result = 0;
      // int shift_count = bit_list.size() - 1;
      for (int i = 0; i < bit_list.size(); ++i) {
        result |= bit_list[i] << (bit_list.size() - 1 - i);
      }
      return result;
    }
    bool next() {
      int i = 0;
      while (i < bit_list.size()) {
        bit_list[i] += 1;
        if (bit_list[i] == 1) {
          return true;
        }
        bit_list[i] = 0;
        i += 1;
      }
      return false;
    }
  };
  Number number(bit_count);

  int N = 2;
  while (--bit_count > 0) {
    N *= 2;
  }
  std::vector<int> result(N);
  for (int i = 0; i < N; ++i) {
    result[i] = number.get_number();
    number.next();
  }
  return result;
}

std::vector<int> index_list_less = init_vector(17);
std::vector<int> index_list_more = init_vector(18);



namespace spconv {
namespace py = pybind11;
using namespace pybind11::literals;

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
  
  std::vector<int>* index_list = &index_list_less;
  if (N > index_list->size()) {
    index_list = &index_list_more;
    if (N > index_list->size()) {
      std::cout << "Too Many Points!\n";
      N = index_list->size();
    }
  }
  int cur = 0;
  int i;
  for (int index = 0; index < N; ++index) {
    i = (*index_list)[cur++];
    while (i >= N) {
      i = (*index_list)[cur++];
    }
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