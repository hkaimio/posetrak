#include "posetrak/tracking/trajectory_stream.hpp"

namespace posetrak {

BatchTrajectoryStream::BatchTrajectoryStream(std::vector<SmoothedFrame> frames,
                                             ForwardKinematics& fk, std::string joint_name)
    : frames_(std::move(frames)), fk_(fk), joint_name_(std::move(joint_name)) {}

std::optional<FreeflyerPose> BatchTrajectoryStream::next() {
    if (next_index_ >= frames_.size()) {
        return std::nullopt;
    }
    SmoothedFrame const& frame = frames_[next_index_++];
    fk_.compute(frame.state);
    auto [position, orientation] = fk_.world_transform(joint_name_);
    return FreeflyerPose{frame.timestamp, position, orientation};
}

}  // namespace posetrak
