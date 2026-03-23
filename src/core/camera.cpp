#include "posetrak/core/camera.hpp"

#include <fmt/core.h>

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace posetrak {

// Intrinsics implementation

nlohmann::json Intrinsics::to_json() const {
    nlohmann::json j;
    j["fx"] = fx;
    j["fy"] = fy;
    j["cx"] = cx;
    j["cy"] = cy;
    j["width"] = width;
    j["height"] = height;

    j["distortion_model"] = (model == DistortionModel::BrownConrady) ? "brown_conrady" : "fisheye";
    j["distortion_coeffs"] = distortion_coeffs;

    return j;
}

Intrinsics Intrinsics::from_json(nlohmann::json const& j) {
    Intrinsics intr;
    intr.fx = j.at("fx").get<double>();
    intr.fy = j.at("fy").get<double>();
    intr.cx = j.at("cx").get<double>();
    intr.cy = j.at("cy").get<double>();
    intr.width = j.at("width").get<int>();
    intr.height = j.at("height").get<int>();

    std::string const model_str = j.value("distortion_model", "brown_conrady");
    intr.model =
        (model_str == "fisheye") ? DistortionModel::Fisheye : DistortionModel::BrownConrady;

    intr.distortion_coeffs = j.value("distortion_coeffs", std::vector<double>{0, 0, 0, 0, 0});

    return intr;
}

// Extrinsics implementation

nlohmann::json Extrinsics::to_json() const {
    nlohmann::json j;
    j["position"] = {position.x(), position.y(), position.z()};
    j["orientation"] = {orientation.w(), orientation.x(), orientation.y(), orientation.z()};
    return j;
}

Extrinsics Extrinsics::from_json(nlohmann::json const& j) {
    Extrinsics extr;

    auto const& pos = j.at("position");
    extr.position =
        Eigen::Vector3d(pos[0].get<double>(), pos[1].get<double>(), pos[2].get<double>());

    auto const& quat = j.at("orientation");
    extr.orientation = Eigen::Quaterniond(quat[0].get<double>(), quat[1].get<double>(),
                                          quat[2].get<double>(), quat[3].get<double>());

    return extr;
}

// SyncPoint implementation

nlohmann::json SyncPoint::to_json() const {
    nlohmann::json j;
    j["frame_idx"] = frame_idx;
    j["timestamp_sec"] = timestamp_sec;
    return j;
}

SyncPoint SyncPoint::from_json(nlohmann::json const& j) {
    SyncPoint sp;
    sp.frame_idx = j.at("frame_idx").get<uint32_t>();
    sp.timestamp_sec = j.at("timestamp_sec").get<double>();
    return sp;
}

// Camera implementation

Camera::Camera(int id, std::string name, Intrinsics const& intrinsics, Extrinsics const& extrinsics,
               double fps, uint32_t start_frame)
    : id_(id),
      name_(std::move(name)),
      intrinsics_(intrinsics),
      extrinsics_(extrinsics),
      fps_(fps),
      start_frame_(start_frame) {
    // Default K_original to K_new so undistort() is a no-op on already-undistorted pixels
    // until set_K_original() is called with the real distorted camera matrix.
    K_original_ = Eigen::Matrix3d::Zero();
    K_original_(0, 0) = intrinsics_.fx;
    K_original_(1, 1) = intrinsics_.fy;
    K_original_(0, 2) = intrinsics_.cx;
    K_original_(1, 2) = intrinsics_.cy;
    K_original_(2, 2) = 1.0;
}

void Camera::set_fps(double fps) {
    if (fps <= 0.0) {
        throw std::invalid_argument("FPS must be positive");
    }
    fps_ = fps;
}

void Camera::set_sync_points(std::vector<SyncPoint> const& points) {
    sync_points_ = points;
    fmt::print("  Camera {}: Loaded {} sync points\n", name_.c_str(), points.size());
    // Sort by frame index for binary search
    std::sort(sync_points_.begin(), sync_points_.end(),
              [](SyncPoint const& a, SyncPoint const& b) { return a.frame_idx < b.frame_idx; });
    for (auto const& sp : sync_points_) {
        fmt::print("    Frame {} -> Time {:.6f} sec\n", sp.frame_idx, sp.timestamp_sec);
    }
}

double Camera::get_timestamp(uint32_t frame_idx) const {
    if (sync_points_.empty()) {
        // Fallback: uniform frame rate
        return static_cast<double>(frame_idx - start_frame_) / fps_;
    }

    // Find bracketing sync points
    auto it =
        std::lower_bound(sync_points_.begin(), sync_points_.end(), frame_idx,
                         [](SyncPoint const& sp, uint32_t idx) { return sp.frame_idx < idx; });

    if (it == sync_points_.begin()) {
        // Before first sync point: extrapolate backward
        // Use signed arithmetic to avoid wraparound when frame_idx < it->frame_idx
        int64_t const frame_diff =
            static_cast<int64_t>(frame_idx) - static_cast<int64_t>(it->frame_idx);
        double const dt = static_cast<double>(frame_diff) / fps_;
        return it->timestamp_sec + dt;
    }

    if (it == sync_points_.end()) {
        // After last sync point: extrapolate forward
        auto const& last = sync_points_.back();
        double const dt = static_cast<double>(frame_idx - last.frame_idx) / fps_;
        return last.timestamp_sec + dt;
    }

    // Linear interpolation between two sync points
    auto const& prev = *(it - 1);
    double const t0 = prev.timestamp_sec;
    double const t1 = it->timestamp_sec;
    uint32_t const f0 = prev.frame_idx;
    uint32_t const f1 = it->frame_idx;

    double const alpha = static_cast<double>(frame_idx - f0) / static_cast<double>(f1 - f0);
    return t0 + alpha * (t1 - t0);
}

std::optional<uint32_t> Camera::get_frame_at_time(double timestamp) const {
    if (sync_points_.empty()) {
        // Fallback: uniform frame rate
        double const t_start = get_timestamp(start_frame_);
        if (timestamp < t_start) {
            return std::nullopt;  // Before first frame
        }
        // Floor: last frame at or before timestamp
        double const frame_offset = (timestamp - t_start) * fps_;
        return start_frame_ + static_cast<uint32_t>(std::floor(frame_offset));
    }

    // Find bracketing sync points
    auto it = std::lower_bound(sync_points_.begin(), sync_points_.end(), timestamp,
                               [](SyncPoint const& sp, double t) { return sp.timestamp_sec < t; });

    if (it == sync_points_.begin()) {
        // Before or at first sync point - extrapolate backward
        if (sync_points_.size() < 2) {
            // Only one sync point: use FPS for extrapolation
            double const dt = timestamp - it->timestamp_sec;
            double const frame_offset = dt * fps_;
            int64_t const frame_int64 = static_cast<int64_t>(it->frame_idx) +
                                        static_cast<int64_t>(std::floor(frame_offset));
            if (frame_int64 < 0) {
                return std::nullopt;  // Before frame 0
            }
            return static_cast<uint32_t>(frame_int64);
        }
        // Multiple sync points: use rate from first two points
        auto const& sp0 = sync_points_[0];
        auto const& sp1 = sync_points_[1];
        double const rate = static_cast<double>(sp1.frame_idx - sp0.frame_idx) /
                            (sp1.timestamp_sec - sp0.timestamp_sec);
        double const frame_offset = (timestamp - sp0.timestamp_sec) * rate;
        int64_t const frame_int64 =
            static_cast<int64_t>(sp0.frame_idx) + static_cast<int64_t>(std::floor(frame_offset));
        if (frame_int64 < 0) {
            return std::nullopt;  // Before frame 0
        }
        return static_cast<uint32_t>(frame_int64);
    }

    if (it == sync_points_.end()) {
        // After last sync point - extrapolate forward
        if (sync_points_.size() < 2) {
            // Only one sync point: use FPS for extrapolation
            auto const& last = sync_points_.back();
            double const dt = timestamp - last.timestamp_sec;
            double const frame_offset = dt * fps_;
            return last.frame_idx + static_cast<uint32_t>(std::floor(frame_offset));
        }
        // Multiple sync points: use rate from last two points
        auto const& sp_prev = sync_points_[sync_points_.size() - 2];
        auto const& sp_last = sync_points_.back();
        double const rate = static_cast<double>(sp_last.frame_idx - sp_prev.frame_idx) /
                            (sp_last.timestamp_sec - sp_prev.timestamp_sec);
        double const frame_offset = (timestamp - sp_last.timestamp_sec) * rate;
        return sp_last.frame_idx + static_cast<uint32_t>(std::floor(frame_offset));
    }

    // Linear interpolation between two sync points
    auto const& prev = *(it - 1);
    double const t0 = prev.timestamp_sec;
    double const t1 = it->timestamp_sec;
    uint32_t const f0 = prev.frame_idx;
    uint32_t const f1 = it->frame_idx;

    double const alpha = (timestamp - t0) / (t1 - t0);
    double const frame_float = f0 + alpha * static_cast<double>(f1 - f0);
    return static_cast<uint32_t>(std::floor(frame_float));
}

std::optional<Eigen::Vector2d> Camera::project(Eigen::Vector3d const& point_world) const {
    // Transform to camera frame
    Eigen::Vector3d const point_cam = world_to_camera(point_world);

    // Check if point is behind camera
    if (point_cam.z() <= 0.0) {
        return std::nullopt;  // Behind camera
    }

    // Normalize to image plane
    Eigen::Vector2d point_norm(point_cam.x() / point_cam.z(), point_cam.y() / point_cam.z());

    // Apply distortion
    Eigen::Vector2d const point_dist = apply_distortion(point_norm);

    // Apply intrinsics
    double const u = intrinsics_.fx * point_dist.x() + intrinsics_.cx;
    double const v = intrinsics_.fy * point_dist.y() + intrinsics_.cy;

    // Check image bounds
    if (u < 0.0 || u >= intrinsics_.width || v < 0.0 || v >= intrinsics_.height) {
        return std::nullopt;  // Out of bounds
    }

    return Eigen::Vector2d(u, v);
}

std::optional<Eigen::Vector2d>
Camera::project_undistorted(Eigen::Vector3d const& point_world) const {
    // Transform to camera frame
    Eigen::Vector3d const point_cam = world_to_camera(point_world);

    // Check if point is behind camera
    if (point_cam.z() <= 0.0) {
        return std::nullopt;  // Behind camera
    }

    // Normalize to image plane (no distortion)
    Eigen::Vector2d point_norm(point_cam.x() / point_cam.z(), point_cam.y() / point_cam.z());

    // Apply intrinsics only (skip distortion)
    double const u = intrinsics_.fx * point_norm.x() + intrinsics_.cx;
    double const v = intrinsics_.fy * point_norm.y() + intrinsics_.cy;

    // Check if projection is within image bounds (match Python behavior)
    if (u < 0.0 || u >= intrinsics_.width || v < 0.0 || v >= intrinsics_.height) {
        return std::nullopt;  // Out of bounds
    }

    return Eigen::Vector2d(u, v);
}

std::vector<Eigen::Vector2d>
Camera::project_batch(std::vector<Eigen::Vector3d> const& points) const {
    std::vector<Eigen::Vector2d> result;
    result.reserve(points.size());

    for (auto const& p : points) {
        auto proj = project(p);
        if (proj) {
            result.push_back(*proj);
        } else {
            // For backward compatibility, push invalid marker
            result.push_back(Eigen::Vector2d(-1.0, -1.0));
        }
    }

    return result;
}

std::vector<Eigen::Vector2d>
Camera::project_batch_undistorted(std::vector<Eigen::Vector3d> const& points) const {
    std::vector<Eigen::Vector2d> result;
    result.reserve(points.size());

    for (auto const& p : points) {
        auto proj = project_undistorted(p);
        if (proj) {
            result.push_back(*proj);
        } else {
            // For backward compatibility, push invalid marker
            result.push_back(Eigen::Vector2d(-1.0, -1.0));
        }
    }

    return result;
}

Eigen::Vector3d Camera::unproject_ray(Eigen::Vector2d const& pixel) const {
    // Remove intrinsics
    Eigen::Vector2d point_dist((pixel.x() - intrinsics_.cx) / intrinsics_.fx,
                               (pixel.y() - intrinsics_.cy) / intrinsics_.fy);

    // Remove distortion
    Eigen::Vector2d const point_norm = remove_distortion(point_dist);

    // Create normalized ray in camera frame (z=1)
    Eigen::Vector3d ray_cam(point_norm.x(), point_norm.y(), 1.0);
    ray_cam.normalize();

    // Transform to world frame
    return extrinsics_.orientation.inverse() * ray_cam;
}

Eigen::Vector3d Camera::unproject(Eigen::Vector2d const& pixel, double depth) const {
    // Remove intrinsics
    Eigen::Vector2d point_dist((pixel.x() - intrinsics_.cx) / intrinsics_.fx,
                               (pixel.y() - intrinsics_.cy) / intrinsics_.fy);

    // Remove distortion
    Eigen::Vector2d const point_norm = remove_distortion(point_dist);

    // Create 3D point in camera frame
    Eigen::Vector3d point_cam(point_norm.x() * depth, point_norm.y() * depth, depth);

    // Transform to world frame
    return camera_to_world(point_cam);
}

Eigen::Vector2d Camera::distort(Eigen::Vector2d const& point_norm) const {
    return apply_distortion(point_norm);
}

void Camera::set_K_original(Eigen::Matrix3d const& K) {
    K_original_ = K;
}

Eigen::Vector2d Camera::undistort(Eigen::Vector2d const& pixel) const {
    // Normalise using K_original (input is in distorted pixel space)
    Eigen::Vector2d point_dist((pixel.x() - K_original_(0, 2)) / K_original_(0, 0),
                               (pixel.y() - K_original_(1, 2)) / K_original_(1, 1));

    // Remove distortion in normalised coordinates
    Eigen::Vector2d const point_norm = remove_distortion(point_dist);

    // Convert to undistorted pixel coordinates using K_new (intrinsics_)
    double const u = intrinsics_.fx * point_norm.x() + intrinsics_.cx;
    double const v = intrinsics_.fy * point_norm.y() + intrinsics_.cy;

    return Eigen::Vector2d(u, v);
}

bool Camera::is_in_bounds(Eigen::Vector2d const& pixel) const {
    return pixel.x() >= 0.0 && pixel.x() < static_cast<double>(intrinsics_.width) &&
           pixel.y() >= 0.0 && pixel.y() < static_cast<double>(intrinsics_.height);
}

Eigen::Matrix<double, 3, 4> Camera::get_projection_matrix() const {
    // Build intrinsic matrix K
    Eigen::Matrix3d K = Eigen::Matrix3d::Zero();
    K(0, 0) = intrinsics_.fx;
    K(1, 1) = intrinsics_.fy;
    K(0, 2) = intrinsics_.cx;
    K(1, 2) = intrinsics_.cy;
    K(2, 2) = 1.0;

    // Get world-to-camera rotation and translation
    Eigen::Matrix3d R = extrinsics_.orientation.toRotationMatrix();
    Eigen::Vector3d t = -R * extrinsics_.position;  // t = -R * C where C is camera center

    // Build [R | t] matrix
    Eigen::Matrix<double, 3, 4> RT;
    RT.block<3, 3>(0, 0) = R;
    RT.col(3) = t;

    // P = K * [R | t]
    return K * RT;
}

nlohmann::json Camera::to_json() const {
    nlohmann::json j;
    j["id"] = id_;
    j["name"] = name_;
    j["intrinsics"] = intrinsics_.to_json();
    j["extrinsics"] = extrinsics_.to_json();
    j["fps"] = fps_;
    j["start_frame"] = start_frame_;

    if (!sync_points_.empty()) {
        j["sync_points"] = nlohmann::json::array();
        for (auto const& sp : sync_points_) {
            j["sync_points"].push_back(sp.to_json());
        }
    }

    return j;
}

Camera Camera::from_json(nlohmann::json const& j) {
    int const id = j.value("id", 0);  // Default to 0 for backward compatibility
    std::string const name = j.at("name").get<std::string>();
    Intrinsics const intrinsics = Intrinsics::from_json(j.at("intrinsics"));
    Extrinsics const extrinsics = Extrinsics::from_json(j.at("extrinsics"));
    double const fps = j.value("fps", 30.0);
    uint32_t const start_frame = j.value("start_frame", 0u);

    Camera cam(id, name, intrinsics, extrinsics, fps, start_frame);

    if (j.contains("sync_points")) {
        std::vector<SyncPoint> sync_points;
        for (auto const& sp_json : j.at("sync_points")) {
            sync_points.push_back(SyncPoint::from_json(sp_json));
        }
        cam.set_sync_points(sync_points);
    }

    return cam;
}

Eigen::Vector3d Camera::world_to_camera(Eigen::Vector3d const& point_world) const {
    return extrinsics_.orientation * (point_world - extrinsics_.position);
}

Eigen::Vector3d Camera::camera_to_world(Eigen::Vector3d const& point_camera) const {
    return extrinsics_.orientation.inverse() * point_camera + extrinsics_.position;
}

Eigen::Vector2d Camera::apply_distortion(Eigen::Vector2d const& point_norm) const {
    if (intrinsics_.model == Intrinsics::DistortionModel::BrownConrady) {
        // Brown-Conrady model (radial + tangential)
        double const x = point_norm.x();
        double const y = point_norm.y();
        double const r2 = x * x + y * y;
        double const r4 = r2 * r2;
        double const r6 = r4 * r2;

        // Extract coefficients (with defaults)
        double const k1 =
            intrinsics_.distortion_coeffs.size() > 0 ? intrinsics_.distortion_coeffs[0] : 0.0;
        double const k2 =
            intrinsics_.distortion_coeffs.size() > 1 ? intrinsics_.distortion_coeffs[1] : 0.0;
        double const p1 =
            intrinsics_.distortion_coeffs.size() > 2 ? intrinsics_.distortion_coeffs[2] : 0.0;
        double const p2 =
            intrinsics_.distortion_coeffs.size() > 3 ? intrinsics_.distortion_coeffs[3] : 0.0;
        double const k3 =
            intrinsics_.distortion_coeffs.size() > 4 ? intrinsics_.distortion_coeffs[4] : 0.0;

        // Radial distortion
        double const radial = 1.0 + k1 * r2 + k2 * r4 + k3 * r6;

        // Tangential distortion
        double const dx = 2.0 * p1 * x * y + p2 * (r2 + 2.0 * x * x);
        double const dy = p1 * (r2 + 2.0 * y * y) + 2.0 * p2 * x * y;

        return Eigen::Vector2d(x * radial + dx, y * radial + dy);
    } else {
        // Fisheye model (OpenCV)
        double const x = point_norm.x();
        double const y = point_norm.y();
        double const r = std::sqrt(x * x + y * y);

        if (r < 1e-8) {
            return point_norm;  // No distortion at center
        }

        // Extract coefficients
        double const k1 =
            intrinsics_.distortion_coeffs.size() > 0 ? intrinsics_.distortion_coeffs[0] : 0.0;
        double const k2 =
            intrinsics_.distortion_coeffs.size() > 1 ? intrinsics_.distortion_coeffs[1] : 0.0;
        double const k3 =
            intrinsics_.distortion_coeffs.size() > 2 ? intrinsics_.distortion_coeffs[2] : 0.0;
        double const k4 =
            intrinsics_.distortion_coeffs.size() > 3 ? intrinsics_.distortion_coeffs[3] : 0.0;

        double const theta = std::atan(r);
        double const theta2 = theta * theta;
        double const theta4 = theta2 * theta2;
        double const theta6 = theta4 * theta2;
        double const theta8 = theta4 * theta4;

        double const theta_d =
            theta * (1.0 + k1 * theta2 + k2 * theta4 + k3 * theta6 + k4 * theta8);

        double const scale = theta_d / r;
        return Eigen::Vector2d(x * scale, y * scale);
    }
}

Eigen::Vector2d Camera::remove_distortion(Eigen::Vector2d const& point_distorted) const {
    if (intrinsics_.model == Intrinsics::DistortionModel::BrownConrady) {
        // Iterative Gauss-Newton to invert distortion
        Eigen::Vector2d point_norm = point_distorted;  // Initial guess

        int const max_iterations = 20;
        double const tolerance = 1e-10;

        for (int i = 0; i < max_iterations; ++i) {
            Eigen::Vector2d const distorted = apply_distortion(point_norm);
            Eigen::Vector2d const error = distorted - point_distorted;

            if (error.squaredNorm() < tolerance) {
                break;
            }

            // Compute Jacobian numerically
            double const eps = 1e-7;
            Eigen::Vector2d const dx = apply_distortion(point_norm + Eigen::Vector2d(eps, 0.0));
            Eigen::Vector2d const dy = apply_distortion(point_norm + Eigen::Vector2d(0.0, eps));

            Eigen::Matrix2d J;
            J.col(0) = (dx - distorted) / eps;
            J.col(1) = (dy - distorted) / eps;

            // Gauss-Newton update
            Eigen::Vector2d const delta = J.colPivHouseholderQr().solve(error);
            point_norm -= delta;
        }

        return point_norm;
    } else {
        // Fisheye model - iterative inversion
        Eigen::Vector2d point_norm = point_distorted;  // Initial guess

        int const max_iterations = 20;
        double const tolerance = 1e-10;

        for (int i = 0; i < max_iterations; ++i) {
            Eigen::Vector2d const distorted = apply_distortion(point_norm);
            Eigen::Vector2d const error = distorted - point_distorted;

            if (error.squaredNorm() < tolerance) {
                break;
            }

            // Numerical Jacobian
            double const eps = 1e-7;
            Eigen::Vector2d const dx = apply_distortion(point_norm + Eigen::Vector2d(eps, 0.0));
            Eigen::Vector2d const dy = apply_distortion(point_norm + Eigen::Vector2d(0.0, eps));

            Eigen::Matrix2d J;
            J.col(0) = (dx - distorted) / eps;
            J.col(1) = (dy - distorted) / eps;

            Eigen::Vector2d const delta = J.colPivHouseholderQr().solve(error);
            point_norm -= delta;
        }

        return point_norm;
    }
}

}  // namespace posetrak
