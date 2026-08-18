// SPDX-FileCopyrightText: 2026 Harri Kaimio
//
// SPDX-License-Identifier: Apache-2.0

/**
 * @file trajectory_stream.hpp
 * @brief Pull-based interface for consuming a completed tracker's smoothed
 * trajectory, one frame's named-joint world transform at a time.
 *
 * Built for the hierarchical body/hand solver (see
 * docs/roadmap/features/hierarchical-solver/hierarchical-solver-design.md,
 * "Parent-trajectory input is a stream"): Stage B (a child filter) needs the
 * parent's freeflyer-boundary joint (e.g. forearm.L) world transform per
 * frame, sourced from the parent's *smoothed* trajectory, not a live
 * per-frame value. TrajectoryStream is the abstract pull interface;
 * BatchTrajectoryStream is the only implementation today, wrapping an
 * already-complete std::vector<SmoothedFrame>. The abstraction exists so a
 * future fixed-lag smoother can supply frames incrementally without
 * changing anything that consumes TrajectoryStream -- only the concrete
 * producer changes.
 */
#pragma once

#include <Eigen/Core>
#include <Eigen/Geometry>

#include "posetrak/filters/rts_smoother.hpp"
#include "posetrak/kinematics/forward_kinematics.hpp"
#include <optional>
#include <string>
#include <vector>

namespace posetrak {

/// @brief World-frame pose of a named joint at one frame of a trajectory.
struct FreeflyerPose {
    double timestamp;
    Eigen::Vector3d position;
    Eigen::Quaterniond orientation;
};

/// @brief Pull-based, sequential source of one named joint's world transform
/// per trajectory frame.
///
/// Consumers call next() until it returns std::nullopt. Frames are always
/// delivered in chronological order; a stream cannot be rewound.
class TrajectoryStream {
   public:
    virtual ~TrajectoryStream() = default;

    /// @brief Advance to and return the next frame's pose, or std::nullopt
    /// once the stream is exhausted.
    virtual std::optional<FreeflyerPose> next() = 0;
};

/// @brief TrajectoryStream over an already-complete smoothed trajectory.
///
/// Wraps a std::vector<SmoothedFrame> (today's full-batch RTS output) and a
/// ForwardKinematics instance built from the same layout that produced that
/// trajectory -- reuse the parent Tracker's own get_fk() rather than
/// constructing a second Pinocchio model. FK is recomputed per frame from
/// the smoothed state to read joint_name's world transform via
/// ForwardKinematics::world_transform().
class BatchTrajectoryStream : public TrajectoryStream {
   public:
    /// @param frames      Smoothed trajectory in chronological order (as
    ///                    returned by Tracker::smooth()).
    /// @param fk          ForwardKinematics for the layout that produced
    ///                    `frames` -- e.g. the parent Tracker's own
    ///                    get_fk(). Not owned; must outlive this stream.
    /// @param joint_name  Name of the joint whose world transform is read
    ///                    out of each frame (the child filter's freeflyer
    ///                    boundary, e.g. "forearm.L").
    BatchTrajectoryStream(std::vector<SmoothedFrame> frames, ForwardKinematics& fk,
                          std::string joint_name);

    std::optional<FreeflyerPose> next() override;

    /// @brief Total number of frames in the stream (for callers that want
    /// to size their own output containers up front).
    size_t size() const { return frames_.size(); }

   private:
    std::vector<SmoothedFrame> frames_;
    ForwardKinematics& fk_;
    std::string joint_name_;
    size_t next_index_ = 0;
};

}  // namespace posetrak
