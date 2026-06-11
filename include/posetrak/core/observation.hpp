#pragma once

#include <Eigen/Core>

#include <nlohmann/json.hpp>

#include <map>
#include <string>
#include <vector>

namespace posetrak {

/// @brief Whether an observation carries an absolute position or a frame-to-frame velocity.
///
/// VELOCITY mode is used for cameras with large systematic position errors (e.g. poor extrinsic
/// calibration). The bias nearly cancels in the frame difference, leaving only the smaller random
/// noise component. The UKF measurement function becomes h(x_t) = project(x_t) - project(x_{t-1})
/// instead of the usual h(x_t) = project(x_t).
enum class MeasurementMode { POSITION, VELOCITY };

/// @brief Single 2D observation of a marker from a camera
struct Observation {
    int camera_id;     ///< Camera identifier
    int marker_id;     ///< Marker identifier
    int frame_idx;     ///< Camera-specific frame number
    double timestamp;  ///< Time in seconds

    Eigen::Vector2d position;            ///< Undistorted pixel coordinates (for UKF)
    Eigen::Vector2d position_distorted;  ///< Original distorted coordinates (for diagnostics)
    double confidence;                   ///< Detection confidence [0, 1] from pose detector

    MeasurementMode mode = MeasurementMode::POSITION;
    Eigen::Vector2d
        prev_position;  ///< Previous frame undistorted pixel; only used when mode == VELOCITY
    double noise_std_override = 0.0;  ///< When > 0, replaces global base noise for this observation

    /// @brief Get measurement noise standard deviation based on confidence
    /// @param base_noise Base noise in pixels
    /// @return Adjusted noise std (higher for low confidence)
    double measurement_noise_std(double base_noise = 5.0) const {
        double effective = (noise_std_override > 0.0) ? noise_std_override : base_noise;
        return effective / std::max(confidence, 0.1);
    }

    /// @brief Serialize to JSON
    nlohmann::json to_json() const;

    /// @brief Deserialize from JSON
    static Observation from_json(nlohmann::json const& j);
};

/// @brief Sequence of observations from a single camera
struct ObservationSequence {
    int camera_id;                          ///< Camera identifier
    std::string camera_name;                ///< Camera name
    std::vector<Observation> observations;  ///< All observations

    /// @brief Query observations in time range [t_start, t_end)
    /// @param t_start Start time (inclusive)
    /// @param t_end End time (exclusive)
    /// @return Observations within time range
    std::vector<Observation> get_in_range(double t_start, double t_end) const;

    /// @brief Get minimum timestamp in sequence
    /// @return Minimum time or infinity if empty
    double min_time() const;

    /// @brief Get maximum timestamp in sequence
    /// @return Maximum time or -infinity if empty
    double max_time() const;

    /// @brief Get number of observations
    /// @return Observation count
    size_t size() const { return observations.size(); }

    /// @brief Check if sequence is empty
    /// @return True if no observations
    bool empty() const { return observations.empty(); }

    /// @brief Serialize to JSON
    nlohmann::json to_json() const;

    /// @brief Deserialize from JSON
    static ObservationSequence from_json(nlohmann::json const& j);
};

/// @brief Multi-camera observation set for a person
class ObservationSet {
   public:
    /// @brief Construct empty observation set
    /// @param person_id Person identifier
    explicit ObservationSet(int person_id = 0);

    /// @brief Add observation sequence for a camera
    /// @param sequence Observation sequence
    void add_sequence(ObservationSequence const& sequence);

    /// @brief Get sequence by camera name
    /// @param camera_name Camera name
    /// @return Pointer to sequence or nullptr if not found
    ObservationSequence const* get_sequence(std::string const& camera_name) const;

    /// @brief Get all observations in time range across all cameras
    /// @param t_start Start time (inclusive)
    /// @param t_end End time (exclusive)
    /// @return All observations in range
    std::vector<Observation> get_all_in_range(double t_start, double t_end) const;

    /// @brief Get minimum timestamp across all cameras
    /// @return Minimum time or infinity if empty
    double min_time() const;

    /// @brief Get maximum timestamp across all cameras
    /// @return Maximum time or -infinity if empty
    double max_time() const;

    /// @brief Get person identifier
    /// @return Person ID
    int person_id() const { return person_id_; }

    /// @brief Get number of cameras
    /// @return Camera count
    size_t camera_count() const { return sequences_.size(); }

    /// @brief Get total number of observations across all cameras
    /// @return Total observation count
    size_t total_observations() const;

    /// @brief Check if observation set is empty
    /// @return True if no observations
    bool empty() const { return sequences_.empty(); }

    /// @brief Get all camera names
    /// @return Camera names
    std::vector<std::string> camera_names() const;

    /// @brief Get all sequences
    /// @return Map of camera_name -> sequence
    std::map<std::string, ObservationSequence> const& sequences() const { return sequences_; }

    /// @brief Serialize to JSON
    nlohmann::json to_json() const;

    /// @brief Deserialize from JSON
    static ObservationSet from_json(nlohmann::json const& j);

   private:
    int person_id_;                                         ///< Person identifier
    std::map<std::string, ObservationSequence> sequences_;  ///< Camera sequences
};

}  // namespace posetrak
