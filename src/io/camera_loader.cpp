#include <posetrak/io/camera_loader.hpp>

#include <Eigen/Geometry>

#include <toml++/toml.hpp>

#include <cmath>
#include <fstream>
#include <sstream>
#include <stdexcept>

namespace posetrak {

namespace {

// Convert rotation vector (Rodrigues) to quaternion
Eigen::Quaterniond rodrigues_to_quaternion(Eigen::Vector3d const& rvec) {
    double angle = rvec.norm();
    if (angle < 1e-10) {
        return Eigen::Quaterniond::Identity();
    }
    Eigen::Vector3d axis = rvec / angle;
    return Eigen::Quaterniond(Eigen::AngleAxisd(angle, axis));
}

// Parse 3x3 intrinsic matrix from TOML array
void parse_intrinsic_matrix(toml::array const& matrix_arr, double& fx, double& fy, double& cx,
                            double& cy) {
    if (matrix_arr.size() != 3) {
        throw std::runtime_error("Intrinsic matrix must be 3x3");
    }

    // Row 0: [fx, 0, cx]
    auto const* row0 = matrix_arr[0].as_array();
    if (!row0 || row0->size() != 3) {
        throw std::runtime_error("Invalid intrinsic matrix row 0");
    }
    fx = row0->at(0).value_or(0.0);
    cx = row0->at(2).value_or(0.0);

    // Row 1: [0, fy, cy]
    auto const* row1 = matrix_arr[1].as_array();
    if (!row1 || row1->size() != 3) {
        throw std::runtime_error("Invalid intrinsic matrix row 1");
    }
    fy = row1->at(1).value_or(0.0);
    cy = row1->at(2).value_or(0.0);
}

// Parse size array [width, height]
void parse_size(toml::array const& size_arr, int& width, int& height) {
    if (size_arr.size() != 2) {
        throw std::runtime_error("Size must be [width, height]");
    }
    width = size_arr[0].value_or(0);
    height = size_arr[1].value_or(0);
}

// Parse distortion coefficients
std::vector<double> parse_distortions(toml::array const& dist_arr) {
    std::vector<double> coeffs;
    coeffs.reserve(dist_arr.size());
    for (auto const& elem : dist_arr) {
        coeffs.push_back(elem.value_or(0.0));
    }
    return coeffs;
}

// Parse 3D vector [x, y, z]
Eigen::Vector3d parse_vector3(toml::array const& arr) {
    if (arr.size() != 3) {
        throw std::runtime_error("Vector must have 3 elements");
    }
    return Eigen::Vector3d(arr[0].value_or(0.0), arr[1].value_or(0.0), arr[2].value_or(0.0));
}

}  // namespace

std::unordered_map<std::string, Camera> load_cameras_from_toml(std::string const& filepath) {
    // Parse TOML file
    toml::table config;
    try {
        config = toml::parse_file(filepath);
    } catch (toml::parse_error const& err) {
        std::ostringstream oss;
        oss << "Failed to parse TOML file '" << filepath << "': " << err.description();
        throw std::runtime_error(oss.str());
    }

    std::unordered_map<std::string, Camera> cameras;

    // Iterate over all tables (cameras)
    for (auto const& [key, value] : config) {
        std::string section_name(key.str());

        // Skip metadata section
        if (section_name == "metadata") {
            continue;
        }

        auto const* cam_table = value.as_table();
        if (!cam_table) {
            continue;  // Skip non-table entries
        }

        try {
            // Required fields
            auto name_opt = (*cam_table)["name"].value<std::string>();
            auto size_arr = (*cam_table)["size"].as_array();
            auto matrix_arr = (*cam_table)["matrix"].as_array();
            auto dist_arr = (*cam_table)["distortions"].as_array();
            auto rot_arr = (*cam_table)["rotation"].as_array();
            auto trans_arr = (*cam_table)["translation"].as_array();
            auto fisheye_opt = (*cam_table)["fisheye"].value<bool>();

            // Validate required fields
            if (!name_opt || !size_arr || !matrix_arr || !dist_arr || !rot_arr || !trans_arr ||
                !fisheye_opt) {
                throw std::runtime_error("Missing required fields in camera section");
            }

            std::string cam_name = *name_opt;

            // Parse intrinsics
            double fx, fy, cx, cy;
            parse_intrinsic_matrix(*matrix_arr, fx, fy, cx, cy);

            int width, height;
            parse_size(*size_arr, width, height);

            std::vector<double> dist_coeffs = parse_distortions(*dist_arr);

            Intrinsics::DistortionModel model = *fisheye_opt
                                                    ? Intrinsics::DistortionModel::Fisheye
                                                    : Intrinsics::DistortionModel::BrownConrady;

            Intrinsics intrinsics{fx, fy, cx, cy, width, height, model, dist_coeffs};

            // Parse extrinsics
            Eigen::Vector3d rvec = parse_vector3(*rot_arr);
            Eigen::Vector3d tvec = parse_vector3(*trans_arr);

            // Convert Rodrigues rotation vector to quaternion
            Eigen::Quaterniond orientation = rodrigues_to_quaternion(rvec);

            Extrinsics extrinsics{tvec, orientation};

            // Create camera (default FPS=30.0, start_frame=0)
            Camera camera(cam_name, intrinsics, extrinsics);

            // Use section name as map key (cam1, cam2, etc.)
            cameras.emplace(section_name, std::move(camera));

        } catch (std::exception const& e) {
            std::ostringstream oss;
            oss << "Failed to parse camera '" << section_name << "': " << e.what();
            throw std::runtime_error(oss.str());
        }
    }

    if (cameras.empty()) {
        throw std::runtime_error("No cameras found in calibration file");
    }

    return cameras;
}

}  // namespace posetrak
