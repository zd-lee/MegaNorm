#include <torch/extension.h>
#include <pico_tree/kd_tree.hpp>
#include <pico_tree/map_traits.hpp>
#include <ATen/Parallel.h>
#include <vector>
#include <algorithm>
#include <mutex>

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

struct ComponentCollector {
  std::mutex mutex;
  std::vector<std::vector<int64_t>> all_components;

  void add_components(std::vector<std::vector<int64_t>>&& comps) {
    std::lock_guard<std::mutex> lock(mutex);
    all_components.insert(
      all_components.end(),
      std::make_move_iterator(comps.begin()),
      std::make_move_iterator(comps.end())
    );
  }
};

} // namespace

std::tuple<torch::Tensor, torch::Tensor> split_patches_connected_cpu(
    torch::Tensor points,
    torch::Tensor patches_2d,
    torch::Tensor patch_sizes,
    int64_t k,
    int64_t min_component_size
) {
  TORCH_CHECK(points.device().is_cpu());
  TORCH_CHECK(patches_2d.device().is_cpu());
  TORCH_CHECK(patch_sizes.device().is_cpu());
  TORCH_CHECK(points.dim() == 2 && points.size(1) == 3);
  TORCH_CHECK(points.dtype() == torch::kFloat32);
  TORCH_CHECK(patches_2d.dtype() == torch::kInt64);
  TORCH_CHECK(patch_sizes.dtype() == torch::kInt64);

  points = points.contiguous();
  patches_2d = patches_2d.contiguous();
  patch_sizes = patch_sizes.contiguous();

  const int64_t N = points.size(0);
  const int64_t P = patches_2d.size(0);

  if (P == 0) {
    return std::make_tuple(
      torch::empty({0, 0}, torch::kInt64),
      torch::empty(0, torch::kInt64)
    );
  }

  float* pts_data = points.data_ptr<float>();
  auto patches_a = patches_2d.accessor<int64_t, 2>();
  auto sizes_a = patch_sizes.accessor<int64_t, 1>();

  ComponentCollector collector;

  at::parallel_for(0, P, 1, [&](int64_t begin, int64_t end) {
    SmallIntSet visited;
    std::vector<int32_t> queue;

    for (int64_t pi = begin; pi < end; pi++) {
      int64_t patch_size = sizes_a[pi];
      if (patch_size == 0) continue;

      // Skip extremely large patches to prevent OOM
      if (patch_size > 5000000) {
        continue;
      }

      // Scale visited set capacity with patch size
      int64_t visited_capacity = next_pow2(std::max<int64_t>(1024, patch_size / 4));
      visited.reset(visited_capacity);

      // Reserve queue space based on patch size (capped at 256K to prevent excessive memory)
      queue.reserve(std::min<size_t>(static_cast<size_t>(patch_size), 262144));

      std::vector<float> patch_points(patch_size * 3);
      std::vector<int64_t> patch_indices(patch_size);

      for (int64_t i = 0; i < patch_size; i++) {
        int64_t idx = patches_a[pi][i];
        TORCH_CHECK(idx >= 0 && idx < N);
        patch_indices[i] = idx;
        patch_points[i * 3 + 0] = pts_data[idx * 3 + 0];
        patch_points[i * 3 + 1] = pts_data[idx * 3 + 1];
        patch_points[i * 3 + 2] = pts_data[idx * 3 + 2];
      }

      pico_tree::space_map<pico_tree::point_map<float, 3>> space(
        patch_points.data(), patch_size
      );
      pico_tree::kd_tree tree(std::ref(space), pico_tree::max_leaf_size_t(10));

      std::vector<std::vector<int>> neighbors(patch_size);
      int64_t k_actual = std::min(k + 1, patch_size);

      for (int64_t i = 0; i < patch_size; i++) {
        std::vector<pico_tree::neighbor<int, float>> knn;
        pico_tree::point_map<float, 3> query(patch_points.data() + i * 3);
        tree.search_knn(query, k_actual, knn);

        neighbors[i].reserve(knn.size() - 1);
        for (size_t j = 1; j < knn.size(); j++) {
          neighbors[i].push_back(knn[j].index);
        }
      }

      std::vector<int> component_id(patch_size, -1);
      std::vector<std::vector<int32_t>> components;

      for (int64_t i = 0; i < patch_size; i++) {
        if (component_id[i] != -1) continue;

        components.push_back({});
        auto& comp = components.back();
        int comp_id = static_cast<int>(components.size() - 1);

        queue.clear();
        queue.push_back(static_cast<int32_t>(i));
        component_id[i] = comp_id;
        comp.push_back(static_cast<int32_t>(i));

        size_t qpos = 0;
        while (qpos < queue.size()) {
          int32_t curr = queue[qpos++];
          for (int nb : neighbors[curr]) {
            if (nb >= 0 && nb < patch_size && component_id[nb] == -1) {
              component_id[nb] = comp_id;
              queue.push_back(static_cast<int32_t>(nb));
              comp.push_back(static_cast<int32_t>(nb));
            }
          }
        }
      }

      std::vector<std::vector<int64_t>> global_components;
      for (const auto& comp : components) {
        if (static_cast<int64_t>(comp.size()) >= min_component_size) {
          std::vector<int64_t> global_comp;
          global_comp.reserve(comp.size());
          for (int32_t local_idx : comp) {
            global_comp.push_back(patch_indices[local_idx]);
          }
          global_components.push_back(std::move(global_comp));
        }
      }

      collector.add_components(std::move(global_components));
    }
  });

  auto& all_comps = collector.all_components;
  if (all_comps.empty()) {
    return std::make_tuple(
      torch::empty({0, 0}, torch::kInt64),
      torch::empty(0, torch::kInt64)
    );
  }

  int64_t Q = static_cast<int64_t>(all_comps.size());
  int64_t M_max = 0;
  for (const auto& c : all_comps) {
    M_max = std::max(M_max, static_cast<int64_t>(c.size()));
  }

  auto packed = torch::full({Q, M_max}, static_cast<int64_t>(-1), torch::kInt64);
  auto sizes = torch::zeros(Q, torch::kInt64);

  auto packed_a = packed.accessor<int64_t, 2>();
  auto sizes_a_out = sizes.accessor<int64_t, 1>();

  for (int64_t i = 0; i < Q; i++) {
    sizes_a_out[i] = static_cast<int64_t>(all_comps[i].size());
    for (size_t j = 0; j < all_comps[i].size(); j++) {
      packed_a[i][j] = all_comps[i][j];
    }
  }

  return std::make_tuple(packed, sizes);
}
