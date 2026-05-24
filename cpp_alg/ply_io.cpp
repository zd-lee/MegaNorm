#include <torch/extension.h>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <stdexcept>
#include <chrono>
#include <tuple>

namespace {

struct PointCloudData {
    std::vector<float> points;    // xyz交错存储，长度 3N
    std::vector<float> normals;   // nx ny nz交错存储，长度 3N
    int64_t num_points;
};

PointCloudData read_ply_binary(const std::string& filepath) {
    std::ifstream file(filepath, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open PLY file: " + filepath);
    }

    std::string line;
    int64_t num_vertices = 0;
    bool is_binary_little_endian = false;
    bool found_xyz = false, found_normals = false, has_colors = false;

    // 解析 ASCII header
    while (std::getline(file, line)) {
        if (line.find("format binary_little_endian") != std::string::npos) {
            is_binary_little_endian = true;
        }
        else if (line.find("element vertex") != std::string::npos) {
            std::istringstream iss(line);
            std::string dummy, dummy2;
            iss >> dummy >> dummy2 >> num_vertices;
        }
        else if (line.find("element face") != std::string::npos) {
            // Skip face data - we only need vertices
            break;
        }
        else if (line.find("property double x") != std::string::npos ||
                 line.find("property float x") != std::string::npos) {
            found_xyz = true;
        }
        else if (line.find("property double nx") != std::string::npos ||
                 line.find("property float nx") != std::string::npos) {
            found_normals = true;
        }
        else if (line.find("property uchar red") != std::string::npos) {
            has_colors = true;
        }
        else if (line.find("end_header") != std::string::npos) {
            break;
        }
    }

    if (!is_binary_little_endian) {
        throw std::runtime_error("Only binary_little_endian PLY format is supported");
    }
    if (!found_xyz || !found_normals) {
        throw std::runtime_error("PLY file must contain x,y,z and nx,ny,nz properties");
    }
    if (num_vertices <= 0) {
        throw std::runtime_error("Invalid number of vertices in PLY header");
    }

    // 读取 binary data
    // 格式：double x, y, z, nx, ny, nz, [uchar r, g, b]
    const size_t bytes_per_vertex = has_colors ? (6 * sizeof(double) + 3) : (6 * sizeof(double));

    PointCloudData data;
    data.num_points = num_vertices;
    data.points.reserve(num_vertices * 3);
    data.normals.reserve(num_vertices * 3);

    // 逐顶点读取
    for (int64_t i = 0; i < num_vertices; i++) {
        double coords[6];
        file.read(reinterpret_cast<char*>(coords), 6 * sizeof(double));

        if (has_colors) {
            unsigned char rgb[3];
            file.read(reinterpret_cast<char*>(rgb), 3);
            // 丢弃颜色数据
        }

        if (!file) {
            throw std::runtime_error("Failed to read PLY binary data at vertex " + std::to_string(i));
        }

        // 转换 double → float
        data.points.push_back(static_cast<float>(coords[0]));  // x
        data.points.push_back(static_cast<float>(coords[1]));  // y
        data.points.push_back(static_cast<float>(coords[2]));  // z
        data.normals.push_back(static_cast<float>(coords[3])); // nx
        data.normals.push_back(static_cast<float>(coords[4])); // ny
        data.normals.push_back(static_cast<float>(coords[5])); // nz
    }

    return data;
}

void write_ply_binary(
    const std::string& filepath,
    const std::vector<float>& points,    // 长度 3N
    const std::vector<float>& normals    // 长度 3N
) {
    if (points.size() != normals.size()) {
        throw std::runtime_error("Points and normals size mismatch");
    }
    if (points.size() % 3 != 0) {
        throw std::runtime_error("Points/normals size must be multiple of 3");
    }

    const int64_t num_vertices = points.size() / 3;

    std::ofstream file(filepath, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to create PLY file: " + filepath);
    }

    // 写入 ASCII header
    file << "ply\n";
    file << "format binary_little_endian 1.0\n";
    file << "comment Created by SegmentOpNet C++ PLY I/O\n";
    file << "element vertex " << num_vertices << "\n";
    file << "property double x\n";
    file << "property double y\n";
    file << "property double z\n";
    file << "property double nx\n";
    file << "property double ny\n";
    file << "property double nz\n";
    file << "end_header\n";

    // 写入 binary data
    // 转换 float → double 并交错写入
    for (int64_t i = 0; i < num_vertices; i++) {
        double data[6] = {
            static_cast<double>(points[i * 3 + 0]),
            static_cast<double>(points[i * 3 + 1]),
            static_cast<double>(points[i * 3 + 2]),
            static_cast<double>(normals[i * 3 + 0]),
            static_cast<double>(normals[i * 3 + 1]),
            static_cast<double>(normals[i * 3 + 2])
        };
        file.write(reinterpret_cast<const char*>(data), sizeof(data));
    }

    if (!file) {
        throw std::runtime_error("Failed to write PLY binary data");
    }
}

} // namespace

// Forward declaration
std::tuple<torch::Tensor, std::vector<double>> extract_largest_component_cpu(
    torch::Tensor points,
    int64_t k
);

std::tuple<torch::Tensor, std::vector<double>> process_ply_file_cpp(
    const std::string& input_path,
    const std::string& output_path,
    int64_t k
) {
    using Clock = std::chrono::high_resolution_clock;
    auto t_total_start = Clock::now();

    // 1. 读取 PLY 文件
    auto t_read_start = Clock::now();
    PointCloudData data = read_ply_binary(input_path);
    auto t_read_end = Clock::now();

    const int64_t N_in = data.num_points;

    // 2. 转换为 torch tensor
    auto points_tensor = torch::from_blob(
        data.points.data(),
        {N_in, 3},
        torch::kFloat32
    ).clone();  // clone to own the data

    // 3. 提取最大连通分量
    auto t_algo_start = Clock::now();
    auto [mask, algo_timing] = extract_largest_component_cpu(points_tensor, k);
    auto t_algo_end = Clock::now();

    auto mask_a = mask.accessor<int64_t, 1>();

    // 4. 过滤 points 和 normals
    auto t_filter_start = Clock::now();

    std::vector<float> filtered_points;
    std::vector<float> filtered_normals;
    int64_t N_out = 0;

    for (int64_t i = 0; i < N_in; i++) {
        if (mask_a[i] == 1) {
            filtered_points.push_back(data.points[i * 3 + 0]);
            filtered_points.push_back(data.points[i * 3 + 1]);
            filtered_points.push_back(data.points[i * 3 + 2]);
            filtered_normals.push_back(data.normals[i * 3 + 0]);
            filtered_normals.push_back(data.normals[i * 3 + 1]);
            filtered_normals.push_back(data.normals[i * 3 + 2]);
            N_out++;
        }
    }

    auto t_filter_end = Clock::now();

    // 5. 写入 PLY 文件
    auto t_write_start = Clock::now();
    write_ply_binary(output_path, filtered_points, filtered_normals);
    auto t_write_end = Clock::now();

    auto t_total_end = Clock::now();

    // 6. 计算统计信息（从 algo_timing 提取）
    // algo_timing = [kdtree, knn, bfs, total, num_components, second_largest]
    int num_components = static_cast<int>(algo_timing[4]);
    int second_largest = static_cast<int>(algo_timing[5]);

    // 创建统计信息 tensor
    auto stats_tensor = torch::tensor(
        {
            static_cast<float>(N_in),
            static_cast<float>(N_out),
            static_cast<float>(num_components),
            static_cast<float>(N_out) / static_cast<float>(N_in),
            static_cast<float>(second_largest)
        },
        torch::kFloat32
    );

    // 7. 计算计时信息
    double t_read = std::chrono::duration<double>(t_read_end - t_read_start).count();
    double t_kdtree = algo_timing[0];  // kdtree_build_time
    double t_knn = algo_timing[1];     // knn_query_time
    double t_bfs = algo_timing[2];     // bfs_time
    double t_filter = std::chrono::duration<double>(t_filter_end - t_filter_start).count();
    double t_write = std::chrono::duration<double>(t_write_end - t_write_start).count();
    double t_total = std::chrono::duration<double>(t_total_end - t_total_start).count();

    std::vector<double> timing = {t_read, t_kdtree, t_knn, t_bfs, t_filter, t_write, t_total};

    return std::make_tuple(stats_tensor, timing);
}
