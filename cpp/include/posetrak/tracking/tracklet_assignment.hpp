// SPDX-FileCopyrightText: 2026 Harri Kaimio
//
// SPDX-License-Identifier: Apache-2.0

/**
 * @file tracklet_assignment.hpp
 * @brief Decisions about which marker slot a dot tracklet is, made outside the tracker.
 *
 * A dot tracklet is one detector-linked dot followed through a camera's frames
 * (db::DotCandidate::tracklet_id). A person reviewing a capture, or an automatic pass
 * that judges whole tracklets against the stored poses of an earlier run, can say
 * which marker a tracklet is for a range of its frames. The tracker's own per-frame
 * assignment sees one frame at a time and rejects a true dot when its pose prediction
 * is a few centimetres off; a decision about the whole tracklet does not depend on the
 * prediction of that frame.
 *
 * A rule applies to the candidates of one (camera, tracklet) whose native video frame is
 * in its range. Of the rules covering a frame the latest wins: an ASSIGN of a marker
 * decides it unless a later FORBID ruled that marker out, REJECT and OTHER exclude the
 * candidate, and a SUGGEST makes its marker cheaper. See TrackletAssignmentModifier
 * (dot_assignment.hpp) for how the tracker uses them.
 */
#pragma once

#include <climits>
#include <string>
#include <unordered_map>
#include <vector>

namespace posetrak {

enum class TrackletRuleKind {
    ASSIGN,   ///< The tracklet is this marker.
    FORBID,   ///< The tracklet is not this marker.
    REJECT,   ///< The tracklet is not a marker of any subject.
    OTHER,    ///< The tracklet is a marker of another subject: no slot of the tracked subject.
    SUGGEST,  ///< The tracklet is probably this marker (an automatic proposal, not a decision).
};

struct TrackletRule {
    TrackletRuleKind kind = TrackletRuleKind::ASSIGN;
    int marker_id = -1;         ///< Index into the skeleton's markers; -1 for REJECT and OTHER.
    int first_frame = INT_MIN;  ///< Inclusive native video frame range the rule covers.
    int last_frame = INT_MAX;

    bool covers(int frame) const { return frame >= first_frame && frame <= last_frame; }
};

/// @brief What the rules of one tracklet say about a frame.
struct TrackletVerdict {
    enum class Kind { NONE, ASSIGNED, EXCLUDED } kind = Kind::NONE;
    int assigned_marker = -1;            ///< Valid for ASSIGNED.
    std::vector<int> forbidden_markers;  ///< Markers ruled out by FORBID rules.
    int suggested_marker = -1;           ///< The marker a SUGGEST rule names, -1 if none.
};

/// @brief Rules for (camera, tracklet) pairs; rules of a tracklet are kept in the order they
/// were added, oldest first, so that a later rule overrides an earlier one.
class TrackletAssignments {
   public:
    void add(int camera_id, int tracklet_id, TrackletRule const& rule);

    /// @brief What the rules say about a candidate; NONE when the tracklet has no rule covering it.
    TrackletVerdict verdict(int camera_id, int tracklet_id, int frame) const;

    std::size_t size() const { return n_rules_; }
    bool empty() const { return n_rules_ == 0; }

   private:
    std::unordered_map<int, std::unordered_map<int, std::vector<TrackletRule>>> rules_;
    std::size_t n_rules_ = 0;
};

/// @brief Result of reading a rule file.
struct TrackletAssignmentFile {
    TrackletAssignments assignments;
    int rows_read = 0;
    int rows_skipped = 0;  ///< Rows naming a camera or marker the run does not have, or malformed.
};

/// @brief Read rules from a CSV file with the columns
/// `camera_label,tracklet_id,first_frame,last_frame,kind,marker_name`.
///
/// first_frame and last_frame are empty for a rule covering the whole tracklet; kind is one of
/// assign, forbid, reject, other, suggest; marker_name is empty for reject and other. A first row
/// whose first field is `camera_label` is a header. A row naming a camera or marker that the
/// run does not have is counted in rows_skipped, not an error: the file may come from a run with
/// a different camera set or skeleton.
///
/// @param path The CSV file.
/// @param camera_ids Camera label -> the id the tracker's candidates carry.
/// @param marker_ids Marker name -> index in the skeleton's markers.
/// @throws std::runtime_error if the file cannot be opened.
TrackletAssignmentFile
load_tracklet_assignments(std::string const& path,
                          std::unordered_map<std::string, int> const& camera_ids,
                          std::unordered_map<std::string, int> const& marker_ids);

}  // namespace posetrak
