#include <torch/extension.h>
#include <ATen/Parallel.h>
#include <cstdint>
#include <vector>
#include <algorithm>
#include <limits>
#include <random>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace {

std::tuple<torch::Tensor, torch::Tensor> pack_patches_for_split(
    const std::vector<std::vector<int64_t>>& patches
) {
    int64_t P = patches.size();
    int64_t M_max = 0;
    for (const auto& p : patches) {
        M_max = std::max(M_max, static_cast<int64_t>(p.size()));
    }

    auto packed = torch::full({P, M_max}, static_cast<int64_t>(-1), torch::kInt64);
    auto sizes = torch::zeros(P, torch::kInt64);

    auto packed_a = packed.accessor<int64_t, 2>();
    auto sizes_a = sizes.accessor<int64_t, 1>();

    for (int64_t i = 0; i < P; i++) {
        sizes_a[i] = patches[i].size();
        for (size_t j = 0; j < patches[i].size(); j++) {
            packed_a[i][j] = patches[i][j];
        }
    }

    return std::make_tuple(packed, sizes);
}

std::vector<torch::Tensor> unpack_split_results(
    torch::Tensor packed,
    torch::Tensor sizes
) {
    std::vector<torch::Tensor> result;
    int64_t Q = sizes.numel();
    auto sizes_a = sizes.accessor<int64_t, 1>();

    for (int64_t i = 0; i < Q; i++) {
        int64_t size = sizes_a[i];
        if (size > 0) {
            result.push_back(packed[i].slice(0, 0, size).clone());
        }
    }

    return result;
}

} // namespace

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
) {
    TORCH_CHECK(points.device().is_cpu(), "points must be CPU tensor");
    TORCH_CHECK(points.dim() == 2 && points.size(1) == 3, "points must be (N, 3)");
    TORCH_CHECK(points.dtype() == torch::kFloat32, "points must be float32");
    TORCH_CHECK(patch_count > 0, "patch_count must be > 0");
    TORCH_CHECK(overlap_count >= 1 && overlap_count <= 10, "overlap_count must be 1-10");
    TORCH_CHECK(k_connectivity > 0, "k_connectivity must be > 0");
    TORCH_CHECK(min_component_size >= 1, "min_component_size must be >= 1");

    points = points.contiguous();

    const int64_t N = points.size(0);

    TORCH_CHECK(patch_count <= N, "patch_count must be <= N");

    if (N == 1) {
        return {torch::tensor({0}, torch::kInt64)};
    }

    if (patch_count >= N) {
        std::vector<torch::Tensor> result;
        for (int64_t i = 0; i < N; i++) {
            result.push_back(torch::tensor({i}, torch::kInt64));
        }
        return result;
    }

    float* pts_data = points.data_ptr<float>();

    std::vector<int64_t> sampled_indices(patch_count);
    std::vector<float> distances(N * overlap_count);
    std::vector<int32_t> assignments(N * overlap_count);
    std::vector<float> current_distances(N);

    const float INF = std::numeric_limits<float>::infinity();
    std::fill(distances.begin(), distances.end(), INF);
    std::fill(assignments.begin(), assignments.end(), -1);

    std::mt19937 rng(42);
    int64_t current_idx = std::uniform_int_distribution<int64_t>(0, N-1)(rng);
    sampled_indices[0] = current_idx;

    for (int64_t iter = 0; iter < patch_count; iter++) {
        float* current_point = pts_data + current_idx * 3;

        #ifdef _OPENMP
        #pragma omp parallel for schedule(static)
        #endif
        for (int64_t i = 0; i < N; i++) {
            float dx = pts_data[i*3+0] - current_point[0];
            float dy = pts_data[i*3+1] - current_point[1];
            float dz = pts_data[i*3+2] - current_point[2];
            current_distances[i] = dx*dx + dy*dy + dz*dz;
        }

        if (overlap_count == 2) {
            #ifdef _OPENMP
            #pragma omp parallel for schedule(static)
            #endif
            for (int64_t i = 0; i < N; i++) {
                float new_dist = current_distances[i];
                int32_t new_idx = static_cast<int32_t>(iter);

                if (new_dist < distances[i*2 + 0]) {
                    distances[i*2 + 1] = distances[i*2 + 0];
                    assignments[i*2 + 1] = assignments[i*2 + 0];
                    distances[i*2 + 0] = new_dist;
                    assignments[i*2 + 0] = new_idx;
                } else if (new_dist < distances[i*2 + 1]) {
                    distances[i*2 + 1] = new_dist;
                    assignments[i*2 + 1] = new_idx;
                }
            }
        } else {
            #ifdef _OPENMP
            #pragma omp parallel for schedule(static)
            #endif
            for (int64_t i = 0; i < N; i++) {
                float new_dist = current_distances[i];
                int32_t new_idx = static_cast<int32_t>(iter);

                for (int64_t k = 0; k < overlap_count; k++) {
                    if (new_dist < distances[i * overlap_count + k]) {
                        for (int64_t j = overlap_count - 1; j > k; j--) {
                            distances[i * overlap_count + j] = distances[i * overlap_count + j - 1];
                            assignments[i * overlap_count + j] = assignments[i * overlap_count + j - 1];
                        }
                        distances[i * overlap_count + k] = new_dist;
                        assignments[i * overlap_count + k] = new_idx;
                        break;
                    }
                }
            }
        }

        if (iter < patch_count - 1) {
            float max_dist = -1.0f;
            int64_t next_idx = 0;

            #ifdef _OPENMP
            #pragma omp parallel
            {
                float local_max = -1.0f;
                int64_t local_idx = 0;

                #pragma omp for nowait
                for (int64_t i = 0; i < N; i++) {
                    float min_dist = distances[i * overlap_count];
                    if (min_dist > local_max) {
                        local_max = min_dist;
                        local_idx = i;
                    }
                }

                #pragma omp critical
                {
                    if (local_max > max_dist) {
                        max_dist = local_max;
                        next_idx = local_idx;
                    }
                }
            }
            #else
            for (int64_t i = 0; i < N; i++) {
                float min_dist = distances[i * overlap_count];
                if (min_dist > max_dist) {
                    max_dist = min_dist;
                    next_idx = i;
                }
            }
            #endif

            current_idx = next_idx;
            sampled_indices[iter + 1] = current_idx;
        }
    }

    std::vector<std::vector<int64_t>> patches(patch_count);
    for (int64_t i = 0; i < N; i++) {
        for (int64_t k = 0; k < overlap_count; k++) {
            int32_t patch_id = assignments[i * overlap_count + k];
            if (patch_id >= 0) {
                patches[patch_id].push_back(i);
            }
        }
    }

    auto [packed, sizes] = pack_patches_for_split(patches);

    auto [connected_packed, connected_sizes] = split_patches_connected_cpu(
        points, packed, sizes, k_connectivity, min_component_size
    );

    return unpack_split_results(connected_packed, connected_sizes);
}
