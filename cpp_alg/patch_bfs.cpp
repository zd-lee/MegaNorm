#include <torch/extension.h>
#include <ATen/Parallel.h>
#include <pico_tree/kd_tree.hpp>
#include <pico_tree/map_traits.hpp>

#include <cstdint>
#include <vector>
#include <algorithm>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace {

static inline uint32_t hash_u32(uint32_t x) {
  x ^= x >> 16;
  x *= 0x7feb352dU;
  x ^= x >> 15;
  x *= 0x846ca68bU;
  x ^= x >> 16;
  return x;
}

static inline int64_t next_pow2(int64_t x) {
  int64_t p = 1;
  while (p < x) {
    p <<= 1;
  }
  return p;
}

struct SmallIntSet {
  std::vector<int32_t> table;
  int32_t mask = 0;

  void reset(int64_t capacity_pow2) {
    table.assign(static_cast<size_t>(capacity_pow2), -1);
    mask = static_cast<int32_t>(capacity_pow2 - 1);
  }

  void clear() {
    std::fill(table.begin(), table.end(), -1);
  }

  bool insert(int32_t key) {
    uint32_t h = hash_u32(static_cast<uint32_t>(key)) & static_cast<uint32_t>(mask);
    while (true) {
      int32_t& slot = table[h];
      if (slot == key) {
        return false;
      }
      if (slot == -1) {
        slot = key;
        return true;
      }
      h = (h + 1) & static_cast<uint32_t>(mask);
    }
  }
};

} // namespace

torch::Tensor bfs_extract_patches_cpu(
    torch::Tensor points,        // (N, 3) float32
    torch::Tensor query_indices, // (N_q,) int32/int64
    int64_t k,                   // KNN parameter
    int64_t num_per_patch) {
  TORCH_CHECK(points.device().is_cpu(), "points must be a CPU tensor");
  TORCH_CHECK(query_indices.device().is_cpu(), "query_indices must be a CPU tensor");
  TORCH_CHECK(points.dim() == 2 && points.size(1) == 3, "points must be (N, 3)");
  TORCH_CHECK(points.dtype() == torch::kFloat32, "points must be float32");
  TORCH_CHECK(query_indices.dim() == 1, "query_indices must be (N_q,)");
  TORCH_CHECK(k > 0, "k must be > 0");
  TORCH_CHECK(num_per_patch > 0, "num_per_patch must be > 0");

  points = points.contiguous();
  query_indices = query_indices.contiguous();

  const int64_t N = points.size(0);
  const int64_t N_q = query_indices.numel();

  if (N_q == 0) {
    return torch::empty({0, num_per_patch}, torch::kInt64);
  }

  const int64_t k_actual = std::min(k, N - 1);

  // Step 1: Build KDTree
  float* pts_data = points.data_ptr<float>();
  pico_tree::space_map<pico_tree::point_map<float, 3>> space(pts_data, N);
  pico_tree::kd_tree tree(std::ref(space), pico_tree::max_leaf_size_t(10));

  // Step 2: Pre-compute KNN neighbors for all points (parallel)
  std::vector<std::vector<int>> neighbors(N);

  #ifdef _OPENMP
  #pragma omp parallel for schedule(dynamic)
  #endif
  for (int64_t i = 0; i < N; i++) {
    std::vector<pico_tree::neighbor<int, float>> knn;
    pico_tree::point_map<float, 3> query(pts_data + i * 3);
    tree.search_knn(query, k_actual + 1, knn);

    neighbors[i].reserve(k_actual);
    for (size_t j = 1; j < knn.size(); j++) {
      neighbors[i].push_back(knn[j].index);
    }
  }

  // Step 3: BFS from each query point (parallel)
  auto out = torch::empty({N_q, num_per_patch}, torch::kInt64);
  auto out_a = out.accessor<int64_t, 2>();

  const bool query_is_i32 = query_indices.scalar_type() == torch::kInt32;
  const bool query_is_i64 = query_indices.scalar_type() == torch::kInt64;
  TORCH_CHECK(query_is_i32 || query_is_i64, "query_indices must be int32 or int64");

  auto get_query = [&](int64_t i) -> int32_t {
    if (query_is_i32) {
      return query_indices.data_ptr<int32_t>()[i];
    }
    return static_cast<int32_t>(query_indices.data_ptr<int64_t>()[i]);
  };

  const int64_t set_capacity = next_pow2(std::max<int64_t>(64, num_per_patch * 4));

  at::parallel_for(0, N_q, 1, [&](int64_t begin, int64_t end) {
    SmallIntSet visited;
    visited.reset(set_capacity);

    std::vector<int32_t> queue;
    queue.reserve(static_cast<size_t>(num_per_patch));

    std::vector<int32_t> patch;
    patch.reserve(static_cast<size_t>(num_per_patch));

    for (int64_t i = begin; i < end; i++) {
      const int32_t start = get_query(i);
      TORCH_CHECK(start >= 0 && start < N, "query index out of range");

      visited.clear();
      queue.clear();
      patch.clear();

      visited.insert(start);
      queue.push_back(start);
      patch.push_back(start);

      size_t qpos = 0;
      while (qpos < queue.size() && static_cast<int64_t>(patch.size()) < num_per_patch) {
        const int32_t current = queue[qpos++];

        for (int nb : neighbors[current]) {
          if (visited.insert(nb)) {
            queue.push_back(nb);
            patch.push_back(nb);
            if (static_cast<int64_t>(patch.size()) >= num_per_patch) {
              break;
            }
          }
        }
      }

      if (static_cast<int64_t>(patch.size()) < num_per_patch) {
        const int32_t pad = patch.empty() ? start : patch.back();
        patch.resize(static_cast<size_t>(num_per_patch), pad);
      }

      for (int64_t j = 0; j < num_per_patch; j++) {
        out_a[i][j] = static_cast<int64_t>(patch[static_cast<size_t>(j)]);
      }
    }
  });

  return out;
}

std::tuple<torch::Tensor, std::vector<double>> extract_largest_component_cpu(
    torch::Tensor points,
    int64_t k
);

std::tuple<torch::Tensor, std::vector<double>> process_ply_file_cpp(
    const std::string& input_path,
    const std::string& output_path,
    int64_t k
);

std::tuple<torch::Tensor, torch::Tensor> split_patches_connected_cpu(
    torch::Tensor points,
    torch::Tensor patches_2d,
    torch::Tensor patch_sizes,
    int64_t k,
    int64_t min_component_size
);

std::vector<torch::Tensor> extract_patches_fps_cpu(
    torch::Tensor points,
    int64_t patch_count,
    int64_t num_per_patch,
    int64_t overlap_count,
    int64_t k_connectivity,
    int64_t min_component_size
);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("bfs_extract_patches_cpu", &bfs_extract_patches_cpu,
        py::arg("points"), py::arg("query_indices"),
        py::arg("k"), py::arg("num_per_patch"));
  m.def("extract_largest_component_cpu", &extract_largest_component_cpu,
        py::arg("points"), py::arg("k"));
  m.def("process_ply_file_cpp", &process_ply_file_cpp,
        py::arg("input_path"), py::arg("output_path"), py::arg("k"));
  m.def("split_patches_connected_cpu", &split_patches_connected_cpu,
        py::arg("points"), py::arg("patches_2d"), py::arg("patch_sizes"),
        py::arg("k"), py::arg("min_component_size"));
  m.def("extract_patches_fps_cpu", &extract_patches_fps_cpu,
        py::arg("points"), py::arg("patch_count"), py::arg("num_per_patch") = 0,
        py::arg("overlap_count") = 2, py::arg("k_connectivity") = 10,
        py::arg("min_component_size") = 5);
}

