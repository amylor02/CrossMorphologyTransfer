import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.widgets import Button, RadioButtons, Slider


def _remove_global_translation(pos_all_joints):
    centered = np.asarray(pos_all_joints, dtype=np.float32).copy()
    centered[..., 0] -= centered[:, 0:1, 0]
    centered[..., 2] -= centered[:, 0:1, 2]
    return centered


def _valid_bones(parents):
    valid = []
    for child_idx, parent_idx in enumerate(np.asarray(parents).tolist()):
        if parent_idx is None:
            continue
        if parent_idx < 0 or parent_idx == child_idx:
            continue
        valid.append((int(parent_idx), int(child_idx)))
    return valid


def _end_site_bones(num_joints, base_joint_count, end_site_parents):
    if end_site_parents is None or num_joints <= base_joint_count:
        return []

    extra_joint_count = min(num_joints - base_joint_count, len(end_site_parents))
    bones = []
    for local_idx in range(extra_joint_count):
        parent_idx = int(end_site_parents[local_idx])
        child_idx = base_joint_count + local_idx
        if parent_idx < 0 or parent_idx >= base_joint_count:
            continue
        bones.append((parent_idx, child_idx))
    return bones


def _prepare_phase_matrix(phase_values, num_frames):
    if phase_values is None:
        return None

    phase = np.asarray(phase_values, dtype=np.float32)
    if phase.ndim == 1:
        phase = phase[:, np.newaxis]

    if phase.shape[0] != num_frames:
        phase = phase[:num_frames]

    return phase


def visualize_contact_motion(pos_all_joints, parents, contact_labels, foot_indices,
                             output_path=None, title="Contact Visualization", fps=30,
                             show=False, max_frames=None, phase_values=None,
                             show_phase_panel=False, end_site_parents=None):
    """
    Visualize a single motion with contact-aware foot highlighting.

    Global translation in X/Z is removed so the animation stays centered.
    Foot joints and their terminal bones are colored differently on contact frames.
    """
    positions = _remove_global_translation(pos_all_joints)
    contacts = (np.asarray(contact_labels) > 0.5).astype(np.uint8)
    if contacts.ndim == 1:
        contacts = contacts[:, np.newaxis]

    if max_frames is not None:
        positions = positions[:max_frames]
        contacts = contacts[:max_frames]

    foot_indices = list(foot_indices)
    num_frames, num_joints, _ = positions.shape
    base_joint_count = len(parents)
    phase_matrix = _prepare_phase_matrix(phase_values, num_frames)
    show_phase_panel = bool(show_phase_panel and phase_matrix is not None)
    phase_window_size = 250
    phase_window_threshold = 350
    bones = _valid_bones(parents)
    end_site_bones = _end_site_bones(num_joints, base_joint_count, end_site_parents)
    bones = bones + end_site_bones
    end_site_indices = list(range(base_joint_count, num_joints)) if num_joints > base_joint_count else []
    foot_map = {joint_idx: local_idx for local_idx, joint_idx in enumerate(foot_indices[:contacts.shape[1]])}

    fig = plt.figure(figsize=(12, 9.1) if show_phase_panel else (8, 9.1))
    if show_phase_panel:
        ax = fig.add_axes([0.05, 0.22, 0.50, 0.70], projection="3d")
        phase_ax = fig.add_axes([0.60, 0.22, 0.35, 0.70])
    else:
        ax = fig.add_axes([0.08, 0.22, 0.84, 0.70], projection="3d")
        phase_ax = None

    ax.set_title(title)
    ax.set_xlabel("X")
    ax.set_ylabel("Z")
    ax.set_zlabel("Y")
    ax.view_init(elev=18, azim=-65)

    x_vals = positions[..., 0]
    z_vals = positions[..., 2]
    y_vals = positions[..., 1]
    radius = max(
        np.max(np.abs(x_vals)),
        np.max(np.abs(z_vals)),
        0.5,
    )
    y_min = float(np.min(y_vals))
    y_max = float(np.max(y_vals))
    if y_max - y_min < 0.5:
        y_max = y_min + 0.5

    ax.set_xlim(-radius, radius)
    ax.set_ylim(-radius, radius)
    ax.set_zlim(y_min, y_max)

    initial = positions[0]
    scatter = ax.scatter(initial[:, 0], initial[:, 2], initial[:, 1], s=18, c="#7a7a7a")
    bone_lines = []
    for parent_idx, child_idx in bones:
        line, = ax.plot(
            [initial[parent_idx, 0], initial[child_idx, 0]],
            [initial[parent_idx, 2], initial[child_idx, 2]],
            [initial[parent_idx, 1], initial[child_idx, 1]],
            color="#b0b0b0",
            linewidth=1.5,
        )
        bone_lines.append((line, child_idx, parent_idx))

    title_text = ax.text2D(0.02, 0.97, "", transform=ax.transAxes)

    phase_lines = []
    phase_cursor = None
    if show_phase_panel:
        phase_ax.set_title("Continuous Phase")
        phase_ax.set_xlabel("Frame")
        phase_ax.set_ylabel("Phase (rad)")
        initial_window_end = min(
            num_frames - 1,
            phase_window_size - 1 if num_frames > phase_window_threshold else num_frames - 1,
        )
        phase_ax.set_xlim(0, max(initial_window_end, 1))
        phase_ax.set_ylim(-0.1, 2.0 * np.pi + 0.1)
        phase_ax.set_yticks([0.0, np.pi, 2.0 * np.pi])
        phase_ax.set_yticklabels(["0", "pi", "2pi"])
        phase_ax.grid(True, alpha=0.3)

        phase_colors = ["#d62728", "#2ca02c", "#ff7f0e", "#9467bd"]
        for foot_idx in range(phase_matrix.shape[1]):
            line, = phase_ax.plot(
                np.arange(phase_matrix.shape[0]),
                phase_matrix[:, foot_idx],
                color=phase_colors[foot_idx % len(phase_colors)],
                linewidth=1.5,
                label=f"foot {foot_idx}",
            )
            phase_lines.append(line)

        phase_cursor = phase_ax.axvline(0, color="#111111", linestyle="--", linewidth=1.5)
        phase_ax.legend(loc="upper right")

    slider_ax = fig.add_axes([0.14, 0.11, 0.52, 0.04])
    prev_ax = fig.add_axes([0.14, 0.04, 0.08, 0.05])
    play_ax = fig.add_axes([0.24, 0.04, 0.10, 0.05])
    next_ax = fig.add_axes([0.36, 0.04, 0.08, 0.05])
    fps_ax = fig.add_axes([0.74, 0.02, 0.18, 0.14])

    frame_slider = Slider(slider_ax, "Frame", 0, num_frames - 1, valinit=0, valstep=1)
    prev_button = Button(prev_ax, "Prev")
    play_button = Button(play_ax, "Pause")
    next_button = Button(next_ax, "Next")
    fps_selector = RadioButtons(fps_ax, ("4 fps", "12 fps", "30 fps"), active={4: 0, 12: 1, 30: 2}.get(fps, 2))

    state = {
        "frame": 0,
        "playing": True,
        "fps": int(fps),
        "slider_updating": False,
    }

    timer = fig.canvas.new_timer(interval=max(1, int(1000 / max(1, state["fps"]))))

    def render_frame(frame_idx, sync_slider=True):
        frame_idx = int(np.clip(frame_idx, 0, num_frames - 1))
        state["frame"] = frame_idx

        frame_pos = positions[frame_idx]
        frame_contacts = contacts[min(frame_idx, contacts.shape[0] - 1)]

        scatter._offsets3d = (frame_pos[:, 0], frame_pos[:, 2], frame_pos[:, 1])

        joint_colors = ["#7a7a7a"] * num_joints
        joint_sizes = np.full(num_joints, 18.0)
        joint_colors[0] = "#1f77b4"
        joint_sizes[0] = 28.0

        for joint_idx in end_site_indices:
            joint_colors[joint_idx] = "#9467bd"
            joint_sizes[joint_idx] = 24.0

        active_feet = []
        for local_idx, joint_idx in enumerate(foot_indices[:contacts.shape[1]]):
            if joint_idx >= num_joints:
                continue
            if frame_contacts[local_idx]:
                joint_colors[joint_idx] = "#d62728"
                joint_sizes[joint_idx] = 48.0
                active_feet.append(str(joint_idx))
            else:
                joint_colors[joint_idx] = "#2ca02c"
                joint_sizes[joint_idx] = 30.0

        scatter.set_color(joint_colors)
        scatter.set_sizes(joint_sizes)

        for line, child_idx, parent_idx in bone_lines:
            line.set_data(
                [frame_pos[parent_idx, 0], frame_pos[child_idx, 0]],
                [frame_pos[parent_idx, 2], frame_pos[child_idx, 2]],
            )
            line.set_3d_properties([frame_pos[parent_idx, 1], frame_pos[child_idx, 1]])
            if child_idx in foot_map and frame_contacts[foot_map[child_idx]]:
                line.set_color("#d62728")
                line.set_linewidth(3.0)
            else:
                line.set_color("#b0b0b0")
                line.set_linewidth(1.5)

        if phase_cursor is not None:
            if num_frames > phase_window_threshold:
                window_idx = frame_idx // phase_window_size
                window_start = window_idx * phase_window_size
                window_end = min(num_frames - 1, window_start + phase_window_size - 1)
                phase_ax.set_xlim(window_start, max(window_end, window_start + 1))
            phase_cursor.set_xdata([frame_idx, frame_idx])

        active_text = ", ".join(active_feet) if active_feet else "none"
        title_text.set_text(f"Frame {frame_idx + 1}/{num_frames} | Contact feet: {active_text}")

        if sync_slider:
            state["slider_updating"] = True
            frame_slider.set_val(frame_idx)
            state["slider_updating"] = False

        fig.canvas.draw_idle()
        return [scatter, title_text] + [line for line, _, _ in bone_lines] + ([phase_cursor] if phase_cursor is not None else [])

    def step_forward(*_args):
        next_frame = (state["frame"] + 1) % num_frames
        render_frame(next_frame)

    timer.add_callback(step_forward)

    def set_playing(is_playing):
        state["playing"] = bool(is_playing)
        play_button.label.set_text("Pause" if state["playing"] else "Play")
        if state["playing"]:
            timer.start()
        else:
            timer.stop()
        fig.canvas.draw_idle()

    def on_slider_change(value):
        if state["slider_updating"]:
            return
        render_frame(int(value), sync_slider=False)

    def on_prev(_event):
        set_playing(False)
        render_frame(max(0, state["frame"] - 1))

    def on_next(_event):
        set_playing(False)
        render_frame(min(num_frames - 1, state["frame"] + 1))

    def on_play_pause(_event):
        set_playing(not state["playing"])

    def on_fps_change(label):
        selected_fps = int(label.split()[0])
        state["fps"] = selected_fps
        timer.interval = max(1, int(1000 / selected_fps))
        if state["playing"]:
            timer.stop()
            timer.start()

    frame_slider.on_changed(on_slider_change)
    prev_button.on_clicked(on_prev)
    next_button.on_clicked(on_next)
    play_button.on_clicked(on_play_pause)
    fps_selector.on_clicked(on_fps_change)

    render_frame(0)
    if state["playing"]:
        timer.start()

    def on_close(_event):
        timer.stop()

    fig.canvas.mpl_connect("close_event", on_close)

    animation = None

    if output_path:
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        try:
            timer.stop()
            animation = FuncAnimation(
                fig,
                lambda frame_idx: render_frame(frame_idx, sync_slider=False),
                frames=num_frames,
                interval=max(1, int(1000 / max(1, state["fps"]))),
                blit=False,
            )
            animation.save(output_path, writer=PillowWriter(fps=fps))
        except Exception as exc:
            print(f"Could not save contact visualization to {output_path}: {exc}")
        finally:
            if animation is not None:
                animation.event_source.stop()
            if state["playing"]:
                timer.start()

    if show:
        plt.show()
    else:
        timer.stop()
        plt.close(fig)

    return {
        "figure": fig,
        "timer": timer,
        "animation": animation,
        "fps_selector": fps_selector,
        "frame_slider": frame_slider,
    }