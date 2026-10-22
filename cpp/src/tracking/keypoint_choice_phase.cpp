// SPDX-FileCopyrightText: 2026 Harri Kaimio
//
// SPDX-License-Identifier: Apache-2.0

#include "posetrak/tracking/keypoint_choice_phase.hpp"

#include <fmt/core.h>

#include "posetrak/tracking/keypoint_window_choice.hpp"
#include "posetrak/tracking/seg_mask.hpp"
#include <algorithm>
#include <chrono>
#include <fstream>
#include <iterator>
#include <set>
#include <sstream>
#include <stdexcept>

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#define PSAPI_VERSION 2
#include <psapi.h>
#include <windows.h>
#elif defined(__linux__)
#include <sys/resource.h>
#endif

namespace posetrak {

namespace {

using Clock = std::chrono::steady_clock;

double seconds_since(Clock::time_point t0) {
    return std::chrono::duration<double>(Clock::now() - t0).count();
}

// Box height in pixels per unit of crop scale (the model input covers 0.75 of a crop that is 1.6
// times the box): the same relation the duplicate merge of the per-step loader uses.
constexpr double kBoxHeightPerCropScale = 160.0;

std::vector<std::string> split(std::string const& text, char sep) {
    std::vector<std::string> out;
    std::string item;
    std::stringstream ss(text);
    while (std::getline(ss, item, sep))
        out.push_back(item);
    if (!text.empty() && text.back() == sep)
        out.emplace_back();
    return out;
}

}  // namespace

double peak_memory_mb() {
#ifdef _WIN32
    PROCESS_MEMORY_COUNTERS counters{};
    if (GetProcessMemoryInfo(GetCurrentProcess(), &counters, sizeof(counters)))
        return static_cast<double>(counters.PeakWorkingSetSize) / (1024.0 * 1024.0);
    return 0.0;
#elif defined(__linux__)
    rusage usage{};
    if (getrusage(RUSAGE_SELF, &usage) == 0)
        return static_cast<double>(usage.ru_maxrss) / 1024.0;  // kilobytes
    return 0.0;
#else
    return 0.0;
#endif
}

KeypointChoiceModel load_keypoint_choice_model(std::string const& path) {
    std::ifstream file(path);
    if (!file)
        throw std::runtime_error("Cannot open keypoint choice model: " + path);
    KeypointChoiceModel model;
    std::string line;
    int number = 0;
    while (std::getline(file, line)) {
        ++number;
        if (!line.empty() && line.back() == '\r')
            line.pop_back();
        if (line.empty() || line[0] == '#')
            continue;
        auto const fields = split(line, ',');
        if (fields.size() >= 4 && fields[0] == "pair") {
            model.pairs.push_back({fields[1], fields[2], std::stod(fields[3])});
        } else if (fields.size() >= 3 && fields[0] == "class") {
            model.keypoint_class[fields[1]] = fields[2];
        } else {
            throw std::runtime_error(fmt::format(
                "{}:{}: expected 'pair,a,b,limit' or 'class,landmark,class'", path, number));
        }
    }
    return model;
}

KeypointChoiceReport run_keypoint_choice_phase(PersonContext& ctx, KeypointChoiceInputs const& in) {
    auto const t_phase = Clock::now();
    KeypointChoiceReport report;
    report.peak_memory_before_mb = peak_memory_mb();
    auto const& cfg = ctx.tracker_config.kp_choice;
    bool const needs_masks = cfg.strict_filter || cfg.mask_term_weight != 0.0;
    if (needs_masks && (in.seg_run_id.empty() || in.seg_mask_label <= 0)) {
        throw std::runtime_error(
            "The strict mask filter and the mask term need the person's masks: pass --seg-run "
            "(and --seg-label if the subject registry does not name the label).");
    }

    // The window holds the tracked range and, on both sides, what the support averages look at.
    double margin_s = cfg.support_window_s;
    if (cfg.mask_term_weight != 0.0)
        margin_s = std::max(margin_s, cfg.mask_term_window_s);
    double const window_start = ctx.start_time - margin_s;
    double const window_end = ctx.end_time + margin_s;

    // ---- 1. load -------------------------------------------------------------------------
    auto t0 = Clock::now();
    SessionReader reader(in.db_path);
    KeypointChoiceData data = reader.load_keypoint_choice_data(
        in.sequence_id, ctx.cameras_by_name, ctx.skeleton, in.person_id, window_start, window_end);
    if (data.candidates.empty() || data.marker_keypoint.empty()) {
        throw std::runtime_error(
            "The tracker config asks for keypoint candidates, but sequence '" + in.sequence_id +
            "' holds none for this skeleton's markers. Run detection with --candidates and "
            "finalise it, or track without candidates.");
    }
    report.load_s = seconds_since(t0);
    report.candidates = static_cast<long>(data.candidates.size());

    int const n_keypoints = static_cast<int>(data.keypoint_landmark.size());
    int const n_cameras = static_cast<int>(data.cameras.size());
    if (n_keypoints > 255 || n_cameras > 255)
        throw std::runtime_error("Keypoint choice: more than 255 keypoints or cameras");
    std::vector<int> keypoint_marker(static_cast<size_t>(n_keypoints), -1);
    for (auto const& [marker, keypoint] : data.marker_keypoint)
        keypoint_marker[static_cast<size_t>(keypoint)] = marker;

    // ---- 2. input of the choice, tracklet decisions --------------------------------------
    kpchoice::Input input;
    input.n_keypoints = n_keypoints;
    input.own_mask_label = in.seg_mask_label;
    input.frames.resize(static_cast<size_t>(n_cameras));
    input.active.assign(static_cast<size_t>(n_cameras), 1);
    for (int c = 0; c < n_cameras; ++c) {
        auto const& cam = data.cameras[static_cast<size_t>(c)];
        for (auto const& f : cam.frames)
            input.frames[static_cast<size_t>(c)].push_back(
                {f.video_frame, f.timestamp, f.crop_scale * kBoxHeightPerCropScale});
        if (std::find(cfg.skip_camera_ids.begin(), cfg.skip_camera_ids.end(), cam.camera_id) !=
            cfg.skip_camera_ids.end())
            input.active[static_cast<size_t>(c)] = 0;
        input.projection.push_back(ctx.cameras_by_id.at(cam.camera_id).get_projection_matrix());
    }
    std::unordered_map<std::string, int> keypoint_of;
    for (int k = 0; k < n_keypoints; ++k)
        if (!data.keypoint_landmark[static_cast<size_t>(k)].empty())
            keypoint_of[data.keypoint_landmark[static_cast<size_t>(k)]] = k;
    input.neighbours.assign(static_cast<size_t>(n_keypoints), {});
    for (auto const& pair : in.model.pairs) {
        auto a = keypoint_of.find(pair.a);
        auto b = keypoint_of.find(pair.b);
        if (a == keypoint_of.end() || b == keypoint_of.end())
            continue;
        input.neighbours[static_cast<size_t>(a->second)].emplace_back(b->second, pair.limit);
        input.neighbours[static_cast<size_t>(b->second)].emplace_back(a->second, pair.limit);
    }
    if (cfg.mask_term_weight != 0.0) {
        std::set<std::string> classes;
        for (auto const& name : split(cfg.mask_term_classes, ','))
            classes.insert(name);
        input.mask_keypoint.assign(static_cast<size_t>(n_keypoints), 0);
        for (int k = 0; k < n_keypoints; ++k) {
            auto it = in.model.keypoint_class.find(data.keypoint_landmark[static_cast<size_t>(k)]);
            if (it != in.model.keypoint_class.end() && classes.count(it->second))
                input.mask_keypoint[static_cast<size_t>(k)] = 1;
        }
    }

    if (cfg.strict_filter) {
        std::set<std::string> classes;
        for (auto const& name : split(cfg.strict_filter_classes, ','))
            classes.insert(name);
        input.filter_keypoint.assign(static_cast<size_t>(n_keypoints), 0);
        for (int k = 0; k < n_keypoints; ++k) {
            auto it = in.model.keypoint_class.find(data.keypoint_landmark[static_cast<size_t>(k)]);
            if (it != in.model.keypoint_class.end() && classes.count(it->second))
                input.filter_keypoint[static_cast<size_t>(k)] = 1;
        }
    }

    bool const has_rules = in.rules != nullptr && !in.rules->empty();
    input.candidates.reserve(data.candidates.size());
    std::set<std::tuple<int, int, int, int>> fixed_seen;
    for (size_t i = 0; i < data.candidates.size(); ++i) {
        auto const& row = data.candidates[i];
        auto const& cam = data.cameras[static_cast<size_t>(row.camera)];
        kpchoice::Candidate c;
        c.x = row.x;
        c.y = row.y;
        c.ux = row.ux;
        c.uy = row.uy;
        c.peak = static_cast<float>(row.peak / in.confidence_scale);
        c.tracklet = row.tracklet_id;
        c.frame = row.frame;
        c.source = static_cast<std::int32_t>(i);
        c.rank = static_cast<std::int16_t>(row.rank);
        c.camera = static_cast<std::uint8_t>(row.camera);
        c.keypoint = static_cast<std::uint8_t>(row.keypoint);
        if (has_rules && row.tracklet_id >= 0) {
            int const video_frame = cam.frames[static_cast<size_t>(row.frame)].video_frame;
            auto const verdict = in.rules->verdict(cam.camera_id, row.tracklet_id, video_frame);
            int const marker = keypoint_marker[static_cast<size_t>(row.keypoint)];
            if (verdict.kind == TrackletVerdict::Kind::EXCLUDED) {
                c.removed = true;
            } else if (verdict.kind == TrackletVerdict::Kind::ASSIGNED) {
                if (verdict.assigned_marker == marker) {
                    if (fixed_seen.emplace(row.camera, row.keypoint, row.frame, row.tracklet_id)
                            .second)
                        input.fixed.push_back(
                            {row.camera, row.keypoint, row.frame, row.tracklet_id});
                } else {
                    c.removed = true;  // the tracklet was decided to be another marker
                }
            }
            if (!c.removed &&
                std::find(verdict.forbidden_markers.begin(), verdict.forbidden_markers.end(),
                          marker) != verdict.forbidden_markers.end())
                c.removed = true;
            report.removed_by_decisions += c.removed;
        }
        input.candidates.push_back(c);
    }

    // ---- 3. mask labels ------------------------------------------------------------------
    if (needs_masks) {
        t0 = Clock::now();
        SessionReader mask_reader(in.db_path);
        size_t i = 0;
        size_t const n = input.candidates.size();
        // Rows come grouped by camera and frame, in that order.
        while (i < n) {
            auto const& first = data.candidates[i];
            size_t j = i;
            while (j < n && data.candidates[j].camera == first.camera &&
                   data.candidates[j].frame == first.frame)
                ++j;
            auto const& cam = data.cameras[static_cast<size_t>(first.camera)];
            int const video_frame = cam.frames[static_cast<size_t>(first.frame)].video_frame;
            DecodedSegMask mask;
            bool have = false;
            auto const svid = in.camera_shot_video_ids.find(cam.camera_id);
            auto const camera = ctx.cameras_by_id.find(cam.camera_id);
            if (svid != in.camera_shot_video_ids.end() && !svid->second.empty() &&
                camera != ctx.cameras_by_id.end()) {
                auto blob = mask_reader.load_seg_mask_blob(in.seg_run_id, svid->second, video_frame,
                                                           in.seg_revision);
                if (blob.has_value()) {
                    mask.native_w = camera->second.intrinsics().width;
                    mask.native_h = camera->second.intrinsics().height;
                    have = decode_seg_mask(*blob, mask);
                }
            }
            for (size_t r = i; r < j; ++r) {
                input.candidates[r].mask_label =
                    have ? static_cast<std::int16_t>(mask_label_at(
                               mask, Eigen::Vector2d(data.candidates[r].x, data.candidates[r].y)))
                         : kpchoice::kMaskUnknown;
            }
            i = j;
        }
        report.masks_s = seconds_since(t0);
    }

    // ---- 4. choice -----------------------------------------------------------------------
    kpchoice::Settings settings;
    settings.support_window_s = cfg.support_window_s;
    settings.mask_window_s = cfg.mask_term_window_s;
    settings.delta = cfg.delta;
    settings.beta = cfg.beta;
    settings.gamma = cfg.gamma;
    settings.mask_weight = cfg.mask_term_weight;
    settings.rounds = cfg.rounds;
    kpchoice::WindowChoice choice(std::move(input), settings);
    t0 = Clock::now();
    choice.flag_consistent();
    report.consensus_s = seconds_since(t0);
    t0 = Clock::now();
    if (cfg.strict_filter)
        choice.apply_strict_filter(cfg.strict_filter_window_frames, cfg.strict_filter_min_fraction);
    else
        choice.count_keypoint_frames();
    choice.run();
    report.choice_s = seconds_since(t0);

    if (!in.dump_path.empty()) {
        std::ofstream dump(in.dump_path);
        dump << "camera_instance_id,video_frame,keypoint,rank,tracklet,consistent,removed,picked\n";
        for (size_t r = 0; r < choice.candidates().size(); ++r) {
            auto const& c = choice.candidates()[r];
            auto const& cam = data.cameras[c.camera];
            bool const picked = choice.input().active[c.camera] &&
                                choice.pick(c.camera, c.keypoint, c.frame) == static_cast<int>(r);
            dump << cam.camera_instance_id << ','
                 << cam.frames[static_cast<size_t>(c.frame)].video_frame << ','
                 << static_cast<int>(c.keypoint) << ',' << c.rank << ',' << c.tracklet << ','
                 << c.consistent << ',' << c.removed << ',' << picked << '\n';
        }
    }

    // ---- 5. stored picks and the observations built from them ----------------------------
    // The picks take the place of the stored body row of every camera that took part, so that
    // everything the loader builds from the row follows them: the positions, the
    // child-minus-parent pairs and the edits applied on top.
    KeypointRowOverrides overrides;
    for (int c = 0; c < n_cameras; ++c) {
        if (!choice.input().active[static_cast<size_t>(c)])
            continue;
        auto const& cam = data.cameras[static_cast<size_t>(c)];
        int const n_frames = static_cast<int>(cam.frames.size());
        KeypointPicks picks;
        picks.camera_instance_id = cam.camera_instance_id;
        picks.n_frames = n_frames;
        picks.n_keypoints = n_keypoints;
        picks.video_frames.reserve(cam.frames.size());
        for (auto const& f : cam.frames)
            picks.video_frames.push_back(f.video_frame);
        picks.tracklet_ids.assign(static_cast<size_t>(n_frames) * n_keypoints, -1);
        picks.ranks.assign(static_cast<size_t>(n_frames) * n_keypoints, -1);
        auto& rows = overrides.rows[cam.camera_instance_id];
        for (int f = 0; f < n_frames; ++f) {
            std::vector<db::Keypoint> row(static_cast<size_t>(n_keypoints));
            for (int k = 0; k < n_keypoints; ++k) {
                int const r = choice.pick(c, k, f);
                if (r < 0)
                    continue;
                auto const& cand = choice.candidates()[static_cast<size_t>(r)];
                auto const& source = data.candidates[static_cast<size_t>(cand.source)];
                size_t const at = static_cast<size_t>(f) * n_keypoints + k;
                picks.tracklet_ids[at] = cand.tracklet >= 0 ? cand.tracklet : -2;
                picks.ranks[at] = static_cast<std::int8_t>(std::min<int>(cand.rank, 127));
                ++picks.n_picked;
                row[static_cast<size_t>(k)] = {source.x, source.y, source.peak};
                report.observations += keypoint_marker[static_cast<size_t>(k)] >= 0 &&
                                       source.peak >= ctx.observation_min_confidence;
            }
            rows[cam.frames[static_cast<size_t>(f)].video_frame] = std::move(row);
        }
        auto const& stats = choice.camera_stats()[static_cast<size_t>(c)];
        picks.n_candidate_keypoint_frames = stats.keypoint_frames;
        picks.n_filter_emptied = stats.emptied;
        report.keypoint_frames += stats.keypoint_frames;
        report.emptied_by_filter += stats.emptied;
        report.picked += picks.n_picked;
        if (ctx.result_writer)
            ctx.result_writer->write_kp_picks(picks);
    }
    auto const& tc = ctx.tracker_config;
    ctx.observations = reader.load_observations(
        in.sequence_id, ctx.cameras_by_name, ctx.skeleton, ctx.observation_min_confidence,
        in.person_id, tc.use_relative_observations, tc.relative_min_confidence, tc.pose_noise_std,
        tc.cross_pair_max_px, tc.cross_pair_max_n, tc.edited_kp_noise_std,
        tc.confidence_threshold_marker_names, tc.confidence_threshold_override,
        ctx.observation_window_start, ctx.observation_window_end, &overrides);

    report.total_s = seconds_since(t_phase);
    report.peak_memory_mb = peak_memory_mb();
    if (!in.quiet) {
        double const share = report.keypoint_frames
                                 ? 100.0 * static_cast<double>(report.emptied_by_filter) /
                                       static_cast<double>(report.keypoint_frames)
                                 : 0.0;
        fmt::print(
            "Keypoint choice (window): {} candidates, {} keypoint-frames, {} picked, {} "
            "observations; {} removed by tracklet decisions\n",
            report.candidates, report.keypoint_frames, report.picked, report.observations,
            report.removed_by_decisions);
        if (cfg.strict_filter) {
            fmt::print("  Strict mask filter emptied {} of {} keypoint-frames ({:.1f}%)\n",
                       report.emptied_by_filter, report.keypoint_frames, share);
        }
        fmt::print(
            "  Phase time {:.1f}s (load {:.1f}, masks {:.1f}, consensus {:.1f}, choice {:.1f}); "
            "peak memory {:.0f} MB ({:.0f} MB before the phase)\n",
            report.total_s, report.load_s, report.masks_s, report.consensus_s, report.choice_s,
            report.peak_memory_mb, report.peak_memory_before_mb);
    }
    return report;
}

}  // namespace posetrak
