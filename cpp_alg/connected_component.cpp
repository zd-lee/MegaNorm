#include <torch/extension.h>
#include <pico_tree/kd_tree.hpp>
#include <pico_tree/map_traits.hpp>
#include <vector>
#include <algorithm>
#include <chrono>
#include <tuple>

#ifdef _OPENMP
#include <omp.h>
#endif

std::tuple<torch::Tensor, std::vector<double>> extract_largest_component_cpu(
    torch::Tensor points,
    int64_t k
) {
    using Clock = std::chrono::high_resolution_clock;
    auto t_start = Clock::now();

    TORCH_CHECK(points.device().is_cpu());
    TORCH_CHECK(points.dim() == 2 && points.size(1) == 3);
    TORCH_CHECK(points.dtype() == torch::kFloat32);

    points = points.contiguous();
    const int64_t N = points.size(0);
    float* pts_data = points.data_ptr<float>();

    pico_tree::space_map<pico_tree::point_map<float, 3>> space(pts_data, N);

    auto t_kdtree_start = Clock::now();
    pico_tree::kd_tree tree(std::ref(space), pico_tree::max_leaf_size_t(10));
    auto t_kdtree_end = Clock::now();

    auto t_knn_start = Clock::now();
    std::vector<std::vector<int>> neighbors(N);

    #ifdef _OPENMP
    #pragma omp parallel for schedule(dynamic)
    #endif
    for (int64_t i = 0; i < N; i++) {
        std::vector<pico_tree::neighbor<int, float>> knn;
        pico_tree::point_map<float, 3> query(pts_data + i * 3);

        tree.search_knn(query, k + 1, knn);

        neighbors[i].reserve(k);
        for (size_t j = 1; j < knn.size(); j++) {
            neighbors[i].push_back(knn[j].index);
        }
    }
    auto t_knn_end = Clock::now();

    auto t_bfs_start = Clock::now();
    std::vector<int> component_id(N, -1);
    std::vector<int> component_sizes;
    std::vector<int> queue;
    queue.reserve(N);

    int next_id = 0;

    for (int64_t i = 0; i < N; i++) {
        if (component_id[i] == -1) {
            queue.clear();
            queue.push_back(i);
            component_id[i] = next_id;
            int size = 1;

            size_t qpos = 0;
            while (qpos < queue.size()) {
                int curr = queue[qpos++];
                for (int nb : neighbors[curr]) {
                    if (nb >= 0 && nb < N && component_id[nb] == -1) {
                        component_id[nb] = next_id;
                        queue.push_back(nb);
                        size++;
                    }
                }
            }

            component_sizes.push_back(size);
            next_id++;
        }
    }

    int largest_id = std::max_element(
        component_sizes.begin(),
        component_sizes.end()
    ) - component_sizes.begin();

    // 计算第二大分量大小
    std::vector<int> sorted_sizes = component_sizes;
    std::sort(sorted_sizes.begin(), sorted_sizes.end(), std::greater<int>());
    int second_largest_size = (sorted_sizes.size() > 1) ? sorted_sizes[1] : 0;

    auto mask = torch::empty({N}, torch::kInt64);
    auto mask_a = mask.accessor<int64_t, 1>();

    for (int64_t i = 0; i < N; i++) {
        mask_a[i] = (component_id[i] == largest_id) ? 1 : 0;
    }
    auto t_bfs_end = Clock::now();

    auto t_end = Clock::now();
    std::vector<double> timing;
    timing.push_back(std::chrono::duration<double>(t_kdtree_end - t_kdtree_start).count());
    timing.push_back(std::chrono::duration<double>(t_knn_end - t_knn_start).count());
    timing.push_back(std::chrono::duration<double>(t_bfs_end - t_bfs_start).count());
    timing.push_back(std::chrono::duration<double>(t_end - t_start).count());
    // 添加统计信息：num_components, second_largest_size
    timing.push_back(static_cast<double>(next_id));  // num_components
    timing.push_back(static_cast<double>(second_largest_size));

    return std::make_tuple(mask, timing);
}
