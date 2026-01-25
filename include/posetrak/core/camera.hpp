#pragma once

#include <Eigen/Core>
#include <Eigen/Geometry>

#include <nlohmann/json.hpp>

#include <string>

namespace posetrak {

/// @brief Camera model with intrinsics, distortion, and extrinsics
///
/// Supports pinhole camera model with radial and tangential distortion.
/// Distortion model follows OpenCV convention:
/// - k1, k2, k3: radial distortion coefficients
/// - p1, p2: tangential distortion coefficients
class Camera {
   public:
    /// @brief Construct camera with intrinsics and optional distortion
    /// @param fx Focal length in x (pixels)
    /// @param fy Focal length in y (pixels)
    /// @param cx Principal point x (pixels)
    /// @param cy Principal point y (pixels)
    /// @param width Image width (pixels)
    /// @param height Image height (pixels)
    /// @param k1 Radial distortion coefficient k1
    /// @param k2 Radial distortion coefficient k2
    /// @param k3 Radial distortion coefficient k3
    /// @param p1 Tangential distortion coefficient p1
    /// @param p2 Tangential distortion coefficient p2
    Camera(double fx, double fy, double cx, double cy, int width, int height, double k1 = 0.0,
           double k2 = 0.0, double k3 = 0.0, double p1 = 0.0, double p2 = 0.0);

    /// @brief Set camera extrinsics (world to camera transform)
    /// @param position Camera position in world frame
    /// @param orientation Camera orientation (quaternion, world to camera)
    void set_extrinsics(Eigen::Vector3d const& position, Eigen::Quaterniond const& orientation);

    /// @brief Get camera position in world frame
    /// @return Camera position
    Eigen::Vector3d position() const { return position_; }

    /// @brief Get camera orientation (world to camera)
    /// @return Orientation quaternion
    Eigen::Quaterniond orientation() const { return orientation_; }

    /// @brief Get focal length x
    /// @return fx in pixels
    double fx() const { return fx_; }

    /// @brief Get focal length y
    /// @return fy in pixels
    double fy() const { return fy_; }

    /// @brief Get principal point x
    /// @return cx in pixels
    double cx() const { return cx_; }

    /// @brief Get principal point y
    /// @return cy in pixels
    double cy() const { return cy_; }

    /// @brief Get image width
    /// @return Width in pixels
    int width() const { return width_; }

    /// @brief Get image height
    /// @return Height in pixels
    int height() const { return height_; }

    /// @brief Project 3D point in world frame to 2D pixel coordinates
    /// @param point_world 3D point in world frame
    /// @return 2D pixel coordinates [u, v]
    Eigen::Vector2d project(Eigen::Vector3d const& point_world) const;

    /// @brief Unproject 2D pixel to 3D ray in world frame
    /// @param pixel 2D pixel coordinates [u, v]
    /// @return Normalized 3D ray direction in world frame
    Eigen::Vector3d unproject_ray(Eigen::Vector2d const& pixel) const;

    /// @brief Unproject 2D pixel with known depth to 3D point in world frame
    /// @param pixel 2D pixel coordinates [u, v]
    /// @param depth Depth in camera frame (positive = in front of camera)
    /// @return 3D point in world frame
    Eigen::Vector3d unproject(Eigen::Vector2d const& pixel, double depth) const;

    /// @brief Apply distortion to normalized image coordinates
    /// @param point_norm Normalized image coordinates [x/z, y/z]
    /// @return Distorted normalized coordinates
    Eigen::Vector2d distort(Eigen::Vector2d const& point_norm) const;

    /// @brief Remove distortion from normalized image coordinates
    /// @param point_distorted Distorted normalized coordinates
    /// @return Undistorted normalized coordinates
    /// @note Uses iterative method (Gauss-Newton)
    Eigen::Vector2d undistort(Eigen::Vector2d const& point_distorted) const;

    /// @brief Check if pixel is within image bounds
    /// @param pixel 2D pixel coordinates
    /// @return True if pixel is within [0, width) x [0, height)
    bool is_in_bounds(Eigen::Vector2d const& pixel) const;

    /// @brief Serialize to JSON
    nlohmann::json to_json() const;

    /// @brief Deserialize from JSON
    static Camera from_json(nlohmann::json const& j);

   private:
    /// @brief Transform point from world to camera frame
    /// @param point_world 3D point in world frame
    /// @return 3D point in camera frame
    Eigen::Vector3d world_to_camera(Eigen::Vector3d const& point_world) const;

    /// @brief Transform point from camera to world frame
    /// @param point_camera 3D point in camera frame
    /// @return 3D point in world frame
    Eigen::Vector3d camera_to_world(Eigen::Vector3d const& point_camera) const;

    // Intrinsics
    double fx_;   ///< Focal length x (pixels)
    double fy_;   ///< Focal length y (pixels)
    double cx_;   ///< Principal point x (pixels)
    double cy_;   ///< Principal point y (pixels)
    int width_;   ///< Image width (pixels)
    int height_;  ///< Image height (pixels)

    // Distortion (OpenCV convention)
    double k1_;  ///< Radial distortion k1
    double k2_;  ///< Radial distortion k2
    double k3_;  ///< Radial distortion k3
    double p1_;  ///< Tangential distortion p1
    double p2_;  ///< Tangential distortion p2

    // Extrinsics
    Eigen::Vector3d position_;        ///< Camera position in world frame
    Eigen::Quaterniond orientation_;  ///< World to camera rotation
};

}  // namespace posetrak
