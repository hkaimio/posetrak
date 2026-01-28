#pragma once

#include <posetrak/core/camera.hpp>
#include <posetrak/core/observation.hpp>

#include <Eigen/Core>

#include <map>
#include <vector>

namespace posetrak {

/// @brief Triangulation result for a single marker
struct TriangulationResult {
    Eigen::Vector3d position;   ///< Triangulated 3D position in world frame
    double reprojection_error;  ///< RMS reprojection error across cameras (pixels)
    int num_cameras;            ///< Number of cameras used for triangulation
    bool success;               ///< Whether triangulation succeeded

    /// @brief Construct failed result
    static TriangulationResult failure() {
        return TriangulationResult{Eigen::Vector3d::Zero(), -1.0, 0, false};
    }
};

/// @brief Triangulate 3D marker positions from 2D observations across multiple cameras
///
/// Supports multiple triangulation methods optimized for different scenarios:
/// - MidPoint: Fast, exact for 2 cameras (mid-point of closest approach)
/// - DLT: Direct Linear Transform, algebraic solution for 2+ cameras
/// - LeastSquares: Non-linear refinement (most accurate, not yet implemented)
///
/// The triangulation is designed for initialization purposes, prioritizing
/// robustness and reasonable accuracy over real-time performance.
class Triangulator {
   public:
    /// @brief Triangulation method
    enum class Method {
        MidPoint,     ///< Mid-point of closest approach (2 cameras, fast)
        DLT,          ///< Direct Linear Transform (algebraic, fast, 2+ cameras)
        LeastSquares  ///< Non-linear least squares (most accurate, slower, not implemented)
    };

    /// @brief Construct triangulator with specified method
    /// @param method Triangulation algorithm to use (default: DLT)
    explicit Triangulator(Method method = Method::DLT);

    /// @brief Triangulate single marker from 2D observations
    ///
    /// Given 2D pixel coordinates (undistorted) from multiple cameras,
    /// reconstructs the 3D position in world frame.
    ///
    /// @param observations 2D pixel positions (undistorted) per camera
    /// @param cameras Camera parameters for each observation (must match observations size)
    /// @param confidences Optional confidence weights [0,1] per observation (empty = equal weight)
    /// @return Triangulation result with 3D position, error metrics, and success flag
    ///
    /// @note Requires at least 2 observations. Returns failure for < 2 cameras.
    /// @note Confidence weighting: higher confidence = more influence on result
    TriangulationResult triangulate(std::vector<Eigen::Vector2d> const& observations,
                                    std::vector<Camera const*> const& cameras,
                                    std::vector<double> const& confidences = {}) const;

    /// @brief Triangulate all visible markers at a specific timestamp
    ///
    /// Processes all markers visible across multiple cameras at the given time,
    /// returning a map of successfully triangulated marker positions.
    ///
    /// @param timestamp Target time to query observations
    /// @param observations All observations across all cameras
    /// @param cameras All available cameras
    /// @param tolerance Time tolerance for matching observations (default: 1e-6)
    /// @return Map from marker_id to triangulation result (only successful triangulations)
    ///
    /// @note Markers visible in < 2 cameras are skipped
    /// @note Uses confidence scores from Observation struct for weighting
    std::map<int, TriangulationResult> triangulate_frame(double timestamp,
                                                         ObservationSet const& observations,
                                                         std::vector<Camera> const& cameras,
                                                         double tolerance = 1e-6) const;

    /// @brief Get current triangulation method
    /// @return Active method
    Method method() const { return method_; }

    /// @brief Set triangulation method
    /// @param method New method to use
    void set_method(Method method) { method_ = method; }

   private:
    Method method_;  ///< Active triangulation method

    /// @brief Triangulate using mid-point of closest approach (2 cameras)
    ///
    /// Computes rays from each camera center through the observation,
    /// then finds the mid-point of the closest approach between rays.
    /// Falls back to simple average if more than 2 cameras provided.
    ///
    /// @param observations 2D pixel positions (undistorted)
    /// @param cameras Camera parameters
    /// @param confidences Confidence weights (not used for mid-point method)
    /// @return Triangulation result
    TriangulationResult triangulate_midpoint(std::vector<Eigen::Vector2d> const& observations,
                                             std::vector<Camera const*> const& cameras,
                                             std::vector<double> const& confidences) const;

    /// @brief Triangulate using Direct Linear Transform (2+ cameras)
    ///
    /// Builds linear system A*X = 0 where X is homogeneous 3D point.
    /// Each observation contributes 2 rows to A. Solves via SVD.
    /// Confidence weighting: scales rows of A by sqrt(confidence).
    ///
    /// @param observations 2D pixel positions (undistorted)
    /// @param cameras Camera parameters
    /// @param confidences Confidence weights (empty = equal weight)
    /// @return Triangulation result
    TriangulationResult triangulate_dlt(std::vector<Eigen::Vector2d> const& observations,
                                        std::vector<Camera const*> const& cameras,
                                        std::vector<double> const& confidences) const;

    /// @brief Triangulate using non-linear least squares (not yet implemented)
    ///
    /// Minimizes sum of squared reprojection errors using optimization.
    /// Uses DLT result as initialization. Most accurate but slower.
    ///
    /// @param observations 2D pixel positions (undistorted)
    /// @param cameras Camera parameters
    /// @param confidences Confidence weights for observation weighting
    /// @return Triangulation result
    /// @throws std::runtime_error Not yet implemented
    TriangulationResult triangulate_least_squares(std::vector<Eigen::Vector2d> const& observations,
                                                  std::vector<Camera const*> const& cameras,
                                                  std::vector<double> const& confidences) const;

    /// @brief Compute RMS reprojection error for a 3D point
    ///
    /// Projects the 3D point to all cameras and computes RMS pixel error.
    ///
    /// @param point_3d 3D point in world frame
    /// @param observations 2D observations (undistorted)
    /// @param cameras Camera parameters
    /// @return RMS reprojection error in pixels
    double compute_reprojection_error(Eigen::Vector3d const& point_3d,
                                      std::vector<Eigen::Vector2d> const& observations,
                                      std::vector<Camera const*> const& cameras) const;
};

}  // namespace posetrak
