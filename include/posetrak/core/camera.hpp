#pragma once

#include <Eigen/Core>
#include <Eigen/Geometry>

#include <nlohmann/json.hpp>

#include <optional>
#include <string>
#include <vector>

namespace posetrak {

/// @brief Camera intrinsic parameters
struct Intrinsics {
    double fx;  ///< Focal length in x (pixels)
    double fy;  ///< Focal length in y (pixels)
    double cx;  ///< Principal point x (pixels)
    double cy;  ///< Principal point y (pixels)

    int width;   ///< Image width (pixels)
    int height;  ///< Image height (pixels)

    /// @brief Distortion model type
    enum class DistortionModel {
        BrownConrady,  ///< Radial + tangential (k1, k2, k3, p1, p2)
        Fisheye        ///< OpenCV fisheye model (k1, k2, k3, k4)
    };

    DistortionModel model;                  ///< Distortion model type
    std::vector<double> distortion_coeffs;  ///< Distortion coefficients

    /// @brief Serialize to JSON
    nlohmann::json to_json() const;

    /// @brief Deserialize from JSON
    static Intrinsics from_json(nlohmann::json const& j);
};

/// @brief Camera extrinsic parameters (world to camera transform)
struct Extrinsics {
    Eigen::Vector3d position;        ///< Camera position in world frame
    Eigen::Quaterniond orientation;  ///< World to camera rotation

    /// @brief Get world to camera transform
    /// @return Affine transform matrix
    Eigen::Affine3d get_transform() const {
        Eigen::Affine3d T = Eigen::Affine3d::Identity();
        T.linear() = orientation.toRotationMatrix();
        T.translation() = position;
        return T;
    }

    /// @brief Serialize to JSON
    nlohmann::json to_json() const;

    /// @brief Deserialize from JSON
    static Extrinsics from_json(nlohmann::json const& j);
};

/// @brief Frame synchronization point for timestamp mapping
struct SyncPoint {
    uint32_t frame_idx;    ///< Frame index (0-based)
    double timestamp_sec;  ///< Timestamp in seconds
    /// @brief Serialize to JSON
    nlohmann::json to_json() const;

    /// @brief Deserialize from JSON
    static SyncPoint from_json(nlohmann::json const& j);
};

/// @brief Camera model with intrinsics, distortion, and extrinsics
///
/// Supports multiple distortion models and frame-to-timestamp conversion.
class Camera {
   public:
    /// @brief Construct camera with name, intrinsics, and extrinsics
    /// @param id Camera numeric identifier
    /// @param name Camera name/identifier
    /// @param intrinsics Camera intrinsic parameters
    /// @param extrinsics Camera extrinsic parameters
    /// @param fps Frame rate (frames per second), used when no sync points available
    /// @param start_frame Starting frame index offset (0-based)
    Camera(int id, std::string name, Intrinsics const& intrinsics, Extrinsics const& extrinsics,
           double fps = 30.0, uint32_t start_frame = 0);

    // --- Accessors ---

    /// @brief Get camera ID
    /// @return Camera ID
    int id() const { return id_; }

    /// @brief Get camera name
    /// @return Camera name
    std::string const& name() const { return name_; }

    /// @brief Get camera intrinsics
    /// @return Intrinsics
    Intrinsics const& intrinsics() const { return intrinsics_; }

    /// @brief Get camera extrinsics
    /// @return Extrinsics
    Extrinsics const& extrinsics() const { return extrinsics_; }

    /// @brief Get frame rate
    /// @return FPS
    double fps() const { return fps_; }

    /// @brief Get start frame offset
    /// @return Start frame index
    uint32_t start_frame() const { return start_frame_; }

    /// @brief Get camera position in world frame
    /// @return Camera position
    Eigen::Vector3d position() const { return extrinsics_.position; }

    /// @brief Get camera orientation (world to camera)
    /// @return Orientation quaternion
    Eigen::Quaterniond orientation() const { return extrinsics_.orientation; }

    // --- Temporal Synchronization ---

    /// @brief Set frame rate
    /// @param fps Frame rate in frames per second
    void set_fps(double fps);

    /// @brief Set synchronization points for frame-to-timestamp conversion
    /// @param points Synchronization points (frame_idx → timestamp)
    void set_sync_points(std::vector<SyncPoint> const& points);

    /// @brief Get timestamp for frame index
    /// @param frame_idx Frame index (0-based)
    /// @return Timestamp in seconds
    /// @note Uses sync points if available, otherwise falls back to uniform FPS
    double get_timestamp(uint32_t frame_idx) const;

    /// @brief Get frame index at given timestamp (inverse lookup)
    /// @param timestamp Timestamp in seconds
    /// @return Last frame index at or before the given timestamp, or std::nullopt if extrapolation
    ///         backward would result in a negative frame index
    /// @note Uses floor semantics: returns the last frame that starts at or before timestamp.
    ///       With sync points: interpolates between points, extrapolates beyond first/last point
    ///       using the rate from the nearest two sync points (or FPS if only one sync point).
    ///       Without sync points: uses uniform FPS from start_frame.
    ///       Returns std::nullopt only if backward extrapolation would result in frame < 0.
    std::optional<uint32_t> get_frame_at_time(double timestamp) const;

    // --- Projection API ---

    // --- Projection API ---

    /// @brief Project 3D point in world frame to 2D pixel coordinates (with distortion)
    /// @param point_world 3D point in world frame
    /// @return 2D pixel coordinates [u, v], or std::nullopt if projection fails (behind camera or
    /// out of bounds)
    std::optional<Eigen::Vector2d> project(Eigen::Vector3d const& point_world) const;

    /// @brief Project 3D point to undistorted pixel coordinates (for UKF)
    /// @param point_world 3D point in world frame
    /// @return Undistorted 2D pixel coordinates, or std::nullopt if projection fails (behind camera
    /// or out of bounds)
    std::optional<Eigen::Vector2d> project_undistorted(Eigen::Vector3d const& point_world) const;

    /// @brief Project multiple points efficiently (with distortion)
    /// @param points 3D points in world frame
    /// @return 2D pixel coordinates
    std::vector<Eigen::Vector2d> project_batch(std::vector<Eigen::Vector3d> const& points) const;

    /// @brief Project multiple points efficiently (undistorted)
    /// @param points 3D points in world frame
    /// @return Undistorted 2D pixel coordinates
    std::vector<Eigen::Vector2d>
    project_batch_undistorted(std::vector<Eigen::Vector3d> const& points) const;

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

    /// @brief Set the original (distorted) camera matrix K_original.
    ///
    /// Must be called when loading cameras from the DB so that undistort()
    /// normalises input pixels correctly using K_original rather than K_new.
    /// If never called, K_original defaults to K_new (undistorted matrix).
    /// @param K 3×3 camera matrix for the original distorted image space
    void set_K_original(Eigen::Matrix3d const& K);

    /// @brief Remove distortion from distorted pixel coordinates → undistorted pixel space.
    ///
    /// Input is assumed to be in distorted pixel space (K_original coordinates).
    /// Output is in undistorted pixel space (K_new / intrinsics_ coordinates).
    /// Uses K_original for input normalisation and the iterative Gauss-Newton method.
    /// @param pixel Distorted pixel coordinates [u, v] in K_original space
    /// @return Undistorted pixel coordinates [u, v] in K_new space
    Eigen::Vector2d undistort(Eigen::Vector2d const& pixel) const;

    /// @brief Check if pixel is within image bounds
    /// @param pixel 2D pixel coordinates
    /// @return True if pixel is within [0, width) x [0, height)
    bool is_in_bounds(Eigen::Vector2d const& pixel) const;

    /// @brief Get 3x4 projection matrix (world to pixel)
    /// @return P = K * [R | t] where K is intrinsics, [R|t] is world-to-camera transform
    Eigen::Matrix<double, 3, 4> get_projection_matrix() const;

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

    /// @brief Apply distortion based on model type
    /// @param point_norm Normalized coordinates
    /// @return Distorted normalized coordinates
    Eigen::Vector2d apply_distortion(Eigen::Vector2d const& point_norm) const;

    /// @brief Remove distortion based on model type
    /// @param point_norm Distorted normalized coordinates
    /// @return Undistorted normalized coordinates
    Eigen::Vector2d remove_distortion(Eigen::Vector2d const& point_norm) const;

    int id_;                              ///< Camera numeric identifier
    std::string name_;                    ///< Camera name/identifier
    Intrinsics intrinsics_;               ///< Intrinsic parameters (K_new / undistorted)
    Extrinsics extrinsics_;               ///< Extrinsic parameters
    double fps_;                          ///< Frame rate (fallback when no sync points)
    int start_frame_;                     ///< Starting frame offset
    std::vector<SyncPoint> sync_points_;  ///< Synchronization points for timestamp conversion
    /// Original (distorted) camera matrix used to normalise distorted input pixels in undistort().
    /// Defaults to K_new (intrinsics_ matrix) when set_K_original() has not been called.
    Eigen::Matrix3d K_original_;
};

}  // namespace posetrak
