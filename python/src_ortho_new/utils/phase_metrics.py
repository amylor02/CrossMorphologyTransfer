import numpy as np


def _ensure_binary_matrix(binary_labels):
    labels = np.asarray(binary_labels)
    if labels.ndim == 1:
        labels = labels[:, np.newaxis]
    return (labels > 0.5).astype(np.uint8)


def _suppress_short_runs(sequence, value, min_length):
    cleaned = np.asarray(sequence, dtype=np.uint8).copy()
    if cleaned.size == 0 or min_length <= 1:
        return cleaned

    run_start = None
    for idx, current in enumerate(cleaned):
        if current == value and run_start is None:
            run_start = idx
        if current != value and run_start is not None:
            if idx - run_start < min_length:
                cleaned[run_start:idx] = 1 - value
            run_start = None

    if run_start is not None and cleaned.size - run_start < min_length:
        cleaned[run_start:] = 1 - value

    return cleaned


def prepare_contact_labels(binary_labels, labels_are_contact=False, min_contact_frames=2, min_swing_frames=2):
    """Convert binary labels to debounced contact labels."""
    labels = _ensure_binary_matrix(binary_labels)
    if not labels_are_contact:
        labels = 1 - labels

    cleaned = np.zeros_like(labels)
    for foot_idx in range(labels.shape[1]):
        foot = labels[:, foot_idx].copy()
        foot = _suppress_short_runs(foot, value=1, min_length=min_contact_frames)
        foot = _suppress_short_runs(foot, value=0, min_length=min_swing_frames)
        foot = _suppress_short_runs(foot, value=1, min_length=min_contact_frames)
        cleaned[:, foot_idx] = foot

    return cleaned


def detect_contact_events(contact_labels):
    """Detect contact-on and contact-off events for each foot."""
    contacts = _ensure_binary_matrix(contact_labels)
    events = []
    for foot_idx in range(contacts.shape[1]):
        foot = contacts[:, foot_idx]
        contact_on = np.where((foot[1:] == 1) & (foot[:-1] == 0))[0] + 1
        contact_off = np.where((foot[1:] == 0) & (foot[:-1] == 1))[0] + 1
        if foot[0] == 1:
            contact_on = np.insert(contact_on, 0, 0)
        events.append(
            {
                "foot_idx": foot_idx,
                "contact_on": contact_on.tolist(),
                "contact_off": contact_off.tolist(),
            }
        )
    return events


def build_continuous_contact_phase(contact_labels):
    """Build a continuous per-frame phase in [0, 2pi) from contact events."""
    contacts = _ensure_binary_matrix(contact_labels)
    phases = np.full(contacts.shape, np.nan, dtype=np.float32)
    events = detect_contact_events(contacts)

    for foot_idx, foot_events in enumerate(events):
        contact_on = np.asarray(foot_events["contact_on"], dtype=np.int64)
        contact_off = np.asarray(foot_events["contact_off"], dtype=np.int64)

        if contact_on.size < 2:
            continue

        for cycle_idx in range(contact_on.size - 1):
            cycle_start = int(contact_on[cycle_idx])
            next_cycle_start = int(contact_on[cycle_idx + 1])
            if next_cycle_start <= cycle_start + 1:
                continue

            cycle_offsets = contact_off[(contact_off > cycle_start) & (contact_off < next_cycle_start)]
            if cycle_offsets.size > 0:
                cycle_off = int(cycle_offsets[0])
            else:
                cycle_off = cycle_start + max(1, (next_cycle_start - cycle_start) // 2)

            cycle_off = min(max(cycle_off, cycle_start + 1), next_cycle_start - 1)

            stance_frames = np.arange(cycle_start, cycle_off)
            swing_frames = np.arange(cycle_off, next_cycle_start)

            if stance_frames.size > 0:
                phases[stance_frames, foot_idx] = np.linspace(
                    0.0, np.pi, num=stance_frames.size, endpoint=False, dtype=np.float32
                )
            if swing_frames.size > 0:
                phases[swing_frames, foot_idx] = np.linspace(
                    np.pi, 2.0 * np.pi, num=swing_frames.size, endpoint=False, dtype=np.float32
                )

    return phases, events


def _count_frames_in_runs(mask, min_run_length=2):
    mask = np.asarray(mask, dtype=np.uint8)
    if mask.size == 0:
        return 0

    run_start = None
    frames_in_runs = 0
    for idx, value in enumerate(mask):
        if value == 1 and run_start is None:
            run_start = idx
        elif value == 0 and run_start is not None:
            run_length = idx - run_start
            if run_length >= min_run_length:
                frames_in_runs += run_length
            run_start = None

    if run_start is not None:
        run_length = mask.size - run_start
        if run_length >= min_run_length:
            frames_in_runs += run_length

    return int(frames_in_runs)


def compute_continuous_phase_error(binary_labels1, binary_labels2, labels_are_contact=False,
                                   min_contact_frames=2, min_swing_frames=2):
    """Compare two motions via contact events and continuous gait phase."""
    contact_labels1 = prepare_contact_labels(
        binary_labels1,
        labels_are_contact=labels_are_contact,
        min_contact_frames=min_contact_frames,
        min_swing_frames=min_swing_frames,
    )
    contact_labels2 = prepare_contact_labels(
        binary_labels2,
        labels_are_contact=labels_are_contact,
        min_contact_frames=min_contact_frames,
        min_swing_frames=min_swing_frames,
    )

    phase1, events1 = build_continuous_contact_phase(contact_labels1)
    phase2, events2 = build_continuous_contact_phase(contact_labels2)

    num_feet = min(contact_labels1.shape[1], contact_labels2.shape[1])
    aggregate_num_feet = min(2, num_feet)
    all_phase_errors = []
    aggregate_disagreement_frames = 0
    aggregate_compared_frames = 0
    aggregate_long_disagreement_frames = 0
    per_foot_phase_error = []
    per_foot_signed_mean = []
    per_foot_on_error = []
    per_foot_off_error = []
    valid_frames_per_foot = []
    per_foot_contact_disagreement_frames = []
    per_foot_contact_disagreement_pct = []
    per_foot_long_contact_disagreement_frames = []
    per_foot_long_contact_disagreement_pct = []

    for foot_idx in range(num_feet):
        valid_len = min(phase1.shape[0], phase2.shape[0])
        phase_1 = phase1[:valid_len, foot_idx]
        phase_2 = phase2[:valid_len, foot_idx]
        contact_1 = contact_labels1[:valid_len, foot_idx]
        contact_2 = contact_labels2[:valid_len, foot_idx]
        valid_mask = np.isfinite(phase_1) & np.isfinite(phase_2)
        valid_frames_per_foot.append(int(valid_mask.sum()))

        disagreement_mask = contact_1 != contact_2
        disagreement_frames = int(np.sum(disagreement_mask))
        long_disagreement_frames = _count_frames_in_runs(disagreement_mask, min_run_length=3)
        compared_frames = int(valid_len)
        per_foot_contact_disagreement_frames.append(disagreement_frames)
        per_foot_contact_disagreement_pct.append(
            float(100.0 * disagreement_frames / compared_frames) if compared_frames > 0 else np.nan
        )
        per_foot_long_contact_disagreement_frames.append(long_disagreement_frames)
        per_foot_long_contact_disagreement_pct.append(
            float(100.0 * long_disagreement_frames / compared_frames) if compared_frames > 0 else np.nan
        )

        if foot_idx < aggregate_num_feet:
            aggregate_disagreement_frames += disagreement_frames
            aggregate_compared_frames += compared_frames
            aggregate_long_disagreement_frames += long_disagreement_frames

        if valid_mask.any():
            wrapped_diff = np.arctan2(
                np.sin(phase_1[valid_mask] - phase_2[valid_mask]),
                np.cos(phase_1[valid_mask] - phase_2[valid_mask]),
            )
            abs_diff = np.abs(wrapped_diff)
            per_foot_phase_error.append(float(np.mean(abs_diff)))
            per_foot_signed_mean.append(
                float(np.arctan2(np.mean(np.sin(wrapped_diff)), np.mean(np.cos(wrapped_diff))))
            )
            if foot_idx < aggregate_num_feet:
                all_phase_errors.extend(abs_diff.tolist())
        else:
            per_foot_phase_error.append(np.nan)
            per_foot_signed_mean.append(np.nan)

        on_1 = np.asarray(events1[foot_idx]["contact_on"], dtype=np.float32)
        on_2 = np.asarray(events2[foot_idx]["contact_on"], dtype=np.float32)
        off_1 = np.asarray(events1[foot_idx]["contact_off"], dtype=np.float32)
        off_2 = np.asarray(events2[foot_idx]["contact_off"], dtype=np.float32)

        if on_1.size > 0 and on_2.size > 0:
            count = min(on_1.size, on_2.size)
            per_foot_on_error.append(float(np.mean(np.abs(on_1[:count] - on_2[:count]))))
        else:
            per_foot_on_error.append(np.nan)

        if off_1.size > 0 and off_2.size > 0:
            count = min(off_1.size, off_2.size)
            per_foot_off_error.append(float(np.mean(np.abs(off_1[:count] - off_2[:count]))))
        else:
            per_foot_off_error.append(np.nan)

    if all_phase_errors:
        avg_phase_error_rad = float(np.mean(all_phase_errors))
        std_phase_error_rad = float(np.std(all_phase_errors))
    else:
        avg_phase_error_rad = np.nan
        std_phase_error_rad = np.nan

    aggregate_contact_disagreement_pct = (
        float(100.0 * aggregate_disagreement_frames / aggregate_compared_frames)
        if aggregate_compared_frames > 0
        else np.nan
    )
    aggregate_long_contact_disagreement_pct = (
        float(100.0 * aggregate_long_disagreement_frames / aggregate_compared_frames)
        if aggregate_compared_frames > 0
        else np.nan
    )

    return {
        "avg_phase_error_rad": avg_phase_error_rad,
        "avg_phase_error_deg": float(np.degrees(avg_phase_error_rad)) if not np.isnan(avg_phase_error_rad) else np.nan,
        "std_phase_error_rad": std_phase_error_rad,
        "std_phase_error_deg": float(np.degrees(std_phase_error_rad)) if not np.isnan(std_phase_error_rad) else np.nan,
        "aggregate_num_feet": aggregate_num_feet,
        "aggregate_contact_disagreement_frames": aggregate_disagreement_frames,
        "aggregate_contact_compared_frames": aggregate_compared_frames,
        "aggregate_contact_disagreement_pct": aggregate_contact_disagreement_pct,
        "aggregate_long_contact_disagreement_frames": aggregate_long_disagreement_frames,
        "aggregate_long_contact_disagreement_pct": aggregate_long_contact_disagreement_pct,
        "per_foot_phase_error_rad": per_foot_phase_error,
        "per_foot_signed_mean_rad": per_foot_signed_mean,
        "per_foot_contact_on_error_frames": per_foot_on_error,
        "per_foot_contact_off_error_frames": per_foot_off_error,
        "per_foot_contact_disagreement_frames": per_foot_contact_disagreement_frames,
        "per_foot_contact_disagreement_pct": per_foot_contact_disagreement_pct,
        "per_foot_long_contact_disagreement_frames": per_foot_long_contact_disagreement_frames,
        "per_foot_long_contact_disagreement_pct": per_foot_long_contact_disagreement_pct,
        "all_phase_errors_rad": all_phase_errors,
        "valid_frames_per_foot": valid_frames_per_foot,
        "contact_labels_1": contact_labels1,
        "contact_labels_2": contact_labels2,
        "phase_1": phase1,
        "phase_2": phase2,
        "events_1": events1,
        "events_2": events2,
    }


def compute_step_timing_difference(binary_labels1, binary_labels2):
    """Measure frame offsets between corresponding step starts."""
    num_feet = binary_labels1.shape[1]
    all_distances = []
    per_foot_distances = []
    step_starts_1_all = []
    step_starts_2_all = []
    for foot_idx in range(num_feet):
        foot1 = binary_labels1[:, foot_idx]
        foot2 = binary_labels2[:, foot_idx]

        step_starts_1 = []
        for i in range(1, len(foot1)):
            if foot1[i] == 1 and foot1[i - 1] == 0:
                step_starts_1.append(i)

        step_starts_2 = []
        for i in range(1, len(foot2)):
            if foot2[i] == 1 and foot2[i - 1] == 0:
                step_starts_2.append(i)

        step_starts_1_all.append(step_starts_1)
        step_starts_2_all.append(step_starts_2)

        foot_distances = []
        for start1 in step_starts_1:
            if len(step_starts_2) == 0:
                continue

            distances_to_all = [abs(start1 - start2) for start2 in step_starts_2]
            min_distance = min(distances_to_all)
            foot_distances.append(min_distance)
            all_distances.append(min_distance)

        if len(foot_distances) > 0:
            per_foot_distances.append(np.mean(foot_distances))
        else:
            per_foot_distances.append(np.nan)

        print(f"Foot {foot_idx}: {len(step_starts_1)} steps in motion 1, {len(step_starts_2)} steps in motion 2")

    if len(all_distances) > 0:
        avg_distance = np.mean(all_distances)
        std = np.std(all_distances)
    else:
        avg_distance = np.nan
        std = np.nan

    return {
        "avg_distance": avg_distance,
        "std": std,
        "per_foot_distances": per_foot_distances,
        "all_distances": all_distances,
        "step_starts_1": step_starts_1_all,
        "step_starts_2": step_starts_2_all,
        "num_steps_1": sum(len(s) for s in step_starts_1_all),
        "num_steps_2": sum(len(s) for s in step_starts_2_all),
    }


def compute_step_timing_difference_radians(binary_labels1, binary_labels2, fps=30):
    """Measure gait timing differences as angular offsets in radians."""
    num_feet = binary_labels1.shape[1]
    all_phase_diffs = []
    per_foot_phase_diffs = []
    all_frame_distances = []
    step_periods_1 = []
    step_periods_2 = []

    for foot_idx in range(num_feet):
        foot1 = binary_labels1[:, foot_idx]
        foot2 = binary_labels2[:, foot_idx]

        step_starts_1 = []
        for i in range(1, len(foot1)):
            if foot1[i] == 1 and foot1[i - 1] == 0:
                step_starts_1.append(i)

        step_starts_2 = []
        for i in range(1, len(foot2)):
            if foot2[i] == 1 and foot2[i - 1] == 0:
                step_starts_2.append(i)

        if len(step_starts_1) < 2 or len(step_starts_2) < 2:
            print(f"Foot {foot_idx}: Insufficient steps for phase analysis")
            per_foot_phase_diffs.append(np.nan)
            step_periods_1.append(np.nan)
            step_periods_2.append(np.nan)
            continue

        periods_1 = np.diff(step_starts_1)
        periods_2 = np.diff(step_starts_2)
        avg_period_1 = np.mean(periods_1)
        avg_period_2 = np.mean(periods_2)

        step_periods_1.append(avg_period_1)
        step_periods_2.append(avg_period_2)

        foot_phase_diffs = []
        for start1 in step_starts_1:
            if len(step_starts_2) == 0:
                continue

            distances_to_all = [abs(start1 - start2) for start2 in step_starts_2]
            min_distance = min(distances_to_all)
            all_frame_distances.append(min_distance)

            avg_period = (avg_period_1 + avg_period_2) / 2.0
            phase_diff = (min_distance / avg_period) * (2 * np.pi)
            phase_diff = np.arctan2(np.sin(phase_diff), np.cos(phase_diff))

            foot_phase_diffs.append(abs(phase_diff))
            all_phase_diffs.append(abs(phase_diff))

        if len(foot_phase_diffs) > 0:
            per_foot_phase_diffs.append(np.mean(foot_phase_diffs))
        else:
            per_foot_phase_diffs.append(np.nan)

        print(
            f"Foot {foot_idx}: {len(step_starts_1)} steps in motion 1, "
            f"{len(step_starts_2)} steps in motion 2, "
            f"avg period: {avg_period_1:.1f} frames (motion 1), {avg_period_2:.1f} frames (motion 2)"
        )

    if len(all_phase_diffs) > 0:
        avg_phase_diff_rad = np.mean(all_phase_diffs)
        std_phase_diff_rad = np.std(all_phase_diffs)
        avg_phase_diff_deg = np.degrees(avg_phase_diff_rad)
    else:
        avg_phase_diff_rad = np.nan
        std_phase_diff_rad = np.nan
        avg_phase_diff_deg = np.nan

    avg_frame_distance = np.mean(all_frame_distances) if len(all_frame_distances) > 0 else np.nan

    return {
        "avg_phase_diff_rad": avg_phase_diff_rad,
        "avg_phase_diff_deg": avg_phase_diff_deg,
        "std_phase_diff_rad": std_phase_diff_rad,
        "std_phase_diff_deg": np.degrees(std_phase_diff_rad) if not np.isnan(std_phase_diff_rad) else np.nan,
        "per_foot_phase_diffs": per_foot_phase_diffs,
        "all_phase_diffs": all_phase_diffs,
        "avg_frame_distance": avg_frame_distance,
        "step_periods_1": step_periods_1,
        "step_periods_2": step_periods_2,
        "num_steps_1": sum(
            len([i for i in range(1, len(binary_labels1[:, f])) if binary_labels1[i, f] == 1 and binary_labels1[i - 1, f] == 0])
            for f in range(num_feet)
        ),
        "num_steps_2": sum(
            len([i for i in range(1, len(binary_labels2[:, f])) if binary_labels2[i, f] == 1 and binary_labels2[i - 1, f] == 0])
            for f in range(num_feet)
        ),
    }


__all__ = [
    "build_continuous_contact_phase",
    "compute_continuous_phase_error",
    "compute_step_timing_difference",
    "compute_step_timing_difference_radians",
    "detect_contact_events",
    "prepare_contact_labels",
]