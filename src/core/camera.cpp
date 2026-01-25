#include "posetrak/core/camera.hpp"

#include <cmath>

namespace posetrak {

Camera::Camera(double fx, double fy, double cx, double cy, int width, int height, double k1,
               double k2, double k3, double p1, double p2)
    : fx_(fx),
      fy_(fy),
      cx_(cx),
      cy_(cy),
      width_(width),
      height_(height),
      k1_(k1),
      k2_(k2),
      k3_(k3),
      p1_(p1),
      p2_(p2),
      position_(Eigen::Vector3d::Zero()),
      orientation_(Eigen::Quaterniond::Identity()) {}

void Camera::set_extrinsics(Eigen::Vector3d const& position,
                            Eigen::Quaterniond const& orientation) {
    position_ = position;
    orientation_ = orientation.normalized();
}

Eigen::Vector2d Camera::project(Eigen::Vector3d const& point_world) const {
    // Transform to camera frame
    Eigen::Vector3d const point_cam = world_to_camera(point_world);

    // Check if point is behind camera
    if (point_cam.z() <= 0.0) {
        return Eigen::Vector2d(-1.0, -1.0);  // Invalid projection
    }

    // Normalize to image plane
    Eigen::Vector2d point_norm(point_cam.x() / point_cam.z(), point_cam.y() / point_cam.z());

    // Apply distortion
    Eigen::Vector2d const point_dist = distort(point_norm);

    // Apply intrinsics
    double const u = fx_ * point_dist.x() + cx_;
    double const v = fy_ * point_dist.y() + cy_;

    return Eigen::Vector2d(u, v);
}

Eigen::Vector3d Camera::unproject_ray(Eigen::Vector2d const& pixel) const {
    // Remove intrinsics
    Eigen::Vector2d point_dist((pixel.x() - cx_) / fx_, (pixel.y() - cy_) / fy_);

    // Remove distortion
    Eigen::Vector2d const point_norm = undistort(point_dist);

    // Create normalized ray in camera frame (z=1)
    Eigen::Vector3d ray_cam(point_norm.x(), point_norm.y(), 1.0);
    ray_cam.normalize();

    // Transform to world frame
    return orientation_.inverse() * ray_cam;
}

Eigen::Vector3d Camera::unproject(Eigen::Vector2d const& pixel, double depth) const {
    // Remove intrinsics
    Eigen::Vector2d point_dist((pixel.x() - cx_) / fx_, (pixel.y() - cy_) / fy_);

    // Remove distortion
    Eigen::Vector2d const point_norm = undistort(point_dist);

    // Create 3D point in camera frame
    Eigen::Vector3d point_cam(point_norm.x() * depth, point_norm.y() * depth, depth);

    // Transform to world frame
    return camera_to_world(point_cam);
}

Eigen::Vector2d Camera::distort(Eigen::Vector2d const& point_norm) const {
    double const x = point_norm.x();
    double const y = point_norm.y();
    double const r2 = x * x + y * y;
    double const r4 = r2 * r2;
    double const r6 = r4 * r2;

    // Radial distortion
    double const radial = 1.0 + k1_ * r2 + k2_ * r4 + k3_ * r6;

    // Tangential distortion
    double const dx = 2.0 * p1_ * x * y + p2_ * (r2 + 2.0 * x * x);
    double const dy = p1_ * (r2 + 2.0 * y * y) + 2.0 * p2_ * x * y;

    return Eigen::Vector2d(x * radial + dx, y * radial + dy);
}

Eigen::Vector2d Camera::undistort(Eigen::Vector2d const& point_distorted) const {
    // Iterative Gauss-Newton to invert distortion
    Eigen::Vector2d point_norm = point_distorted;  // Initial guess

    int const max_iterations = 20;
    double const tolerance = 1e-10;

    for (int i = 0; i < max_iterations; ++i) {
        Eigen::Vector2d const distorted = distort(point_norm);
        Eigen::Vector2d const error = distorted - point_distorted;

        if (error.squaredNorm() < tolerance) {
            break;
        }

        // Compute Jacobian numerically
        double const eps = 1e-7;
        Eigen::Vector2d const dx = distort(point_norm + Eigen::Vector2d(eps, 0.0));
        Eigen::Vector2d const dy = distort(point_norm + Eigen::Vector2d(0.0, eps));

        Eigen::Matrix2d J;
        J.col(0) = (dx - distorted) / eps;
        J.col(1) = (dy - distorted) / eps;

        // Gauss-Newton update
        Eigen::Vector2d const delta = J.colPivHouseholderQr().solve(error);
        point_norm -= delta;
    }

    return point_norm;
}

bool Camera::is_in_bounds(Eigen::Vector2d const& pixel) const {
    return pixel.x() >= 0.0 && pixel.x() < static_cast<double>(width_) && pixel.y() >= 0.0 &&
           pixel.y() < static_cast<double>(height_);
}

nlohmann::json Camera::to_json() const {
    nlohmann::json j;

    // Intrinsics
    j["intrinsics"]["fx"] = fx_;
    j["intrinsics"]["fy"] = fy_;
    j["intrinsics"]["cx"] = cx_;
    j["intrinsics"]["cy"] = cy_;
    j["intrinsics"]["width"] = width_;
    j["intrinsics"]["height"] = height_;

    // Distortion
    j["distortion"]["k1"] = k1_;
    j["distortion"]["k2"] = k2_;
    j["distortion"]["k3"] = k3_;
    j["distortion"]["p1"] = p1_;
    j["distortion"]["p2"] = p2_;

    // Extrinsics
    j["extrinsics"]["position"] = {position_.x(), position_.y(), position_.z()};
    j["extrinsics"]["orientation"] = {orientation_.w(), orientation_.x(), orientation_.y(),
                                      orientation_.z()};

    return j;
}

Camera Camera::from_json(nlohmann::json const& j) {
    // Intrinsics
    auto const& intr = j.at("intrinsics");
    double const fx = intr.at("fx").get<double>();
    double const fy = intr.at("fy").get<double>();
    double const cx = intr.at("cx").get<double>();
    double const cy = intr.at("cy").get<double>();
    int const width = intr.at("width").get<int>();
    int const height = intr.at("height").get<int>();

    // Distortion (optional)
    double k1 = 0.0, k2 = 0.0, k3 = 0.0, p1 = 0.0, p2 = 0.0;
    if (j.contains("distortion")) {
        auto const& dist = j.at("distortion");
        k1 = dist.value("k1", 0.0);
        k2 = dist.value("k2", 0.0);
        k3 = dist.value("k3", 0.0);
        p1 = dist.value("p1", 0.0);
        p2 = dist.value("p2", 0.0);
    }

    Camera cam(fx, fy, cx, cy, width, height, k1, k2, k3, p1, p2);

    // Extrinsics (optional)
    if (j.contains("extrinsics")) {
        auto const& extr = j.at("extrinsics");
        auto const& pos = extr.at("position");
        auto const& quat = extr.at("orientation");

        Eigen::Vector3d position(pos[0].get<double>(), pos[1].get<double>(), pos[2].get<double>());
        Eigen::Quaterniond orientation(quat[0].get<double>(), quat[1].get<double>(),
                                       quat[2].get<double>(), quat[3].get<double>());

        cam.set_extrinsics(position, orientation);
    }

    return cam;
}

Eigen::Vector3d Camera::world_to_camera(Eigen::Vector3d const& point_world) const {
    return orientation_ * (point_world - position_);
}

Eigen::Vector3d Camera::camera_to_world(Eigen::Vector3d const& point_camera) const {
    return orientation_.inverse() * point_camera + position_;
}

}  // namespace posetrak
