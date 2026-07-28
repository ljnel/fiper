"""Dash app: interactive 3D view of predicted action chunks (EE position) over an episode,
with a synced RGB camera view and a time-step slider.

Pick a task + episode; the title shows whether the rollout succeeded or failed.
Generalizes across the 4 position-action tasks (stacking, pretzel, push_t, push_chair);
sorting is a velocity-action task and has no EE position to plot, so it is disabled.

Run:  pixi run python my_nbs/chunk_app.py   then open http://127.0.0.1:8050
"""
import glob
import os
import pickle
import random
import re
from functools import lru_cache

import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.colors import sample_colorscale
from dash import Dash, dcc, html, Input, Output, State, ctx, no_update

DATA_ROOT = os.path.join(os.path.dirname(__file__), "..", "data")
POSITION_TASKS = ["stacking", "pretzel", "push_t", "push_chair"]  # expose action_mappings["pos"]
ALL_TASKS = POSITION_TASKS + ["sorting"]                          # sorting = velocity action, disabled


def unwrap(x):
    """push_t / push_chair store per-robot tensors as a length-1 list; others are plain arrays."""
    return x[0] if isinstance(x, list) else x


def to_displayable(img):
    """Coerce an rgb frame to HWC uint8 (pretzel is CHW float32 [0,1]; others HWC uint8)."""
    img = unwrap(img)
    if img.ndim == 3 and img.shape[0] in (1, 3) and img.shape[-1] not in (1, 3):
        img = np.transpose(img, (1, 2, 0))                  # CHW -> HWC
    if img.dtype != np.uint8:
        img = (np.clip(img, 0, 1) * 255).astype(np.uint8)   # float [0,1] -> uint8
    return img


@lru_cache(maxsize=8192)
def episode_outcome(path):
    """SUCCESS/FAIL. Cheap from the filename (stacking/push_t/push_chair); pretzel names lack the
    marker, so fall back to metadata['successful'] (cached, so the pickle is read at most once)."""
    name = os.path.basename(path)
    if "_s_" in name:
        return "SUCCESS"
    if "_f_" in name:
        return "FAIL"
    with open(path, "rb") as fh:
        return "SUCCESS" if pickle.load(fh)["metadata"]["successful"] else "FAIL"


def list_episodes(task):
    """Glob filenames; index from the name, outcome from filename-or-metadata (see episode_outcome)."""
    eps = []
    for split in ["test", "calibration"]:
        d = os.path.join(DATA_ROOT, task, "rollouts", split)
        for p in sorted(glob.glob(os.path.join(d, "episode_*.pkl"))):
            m = re.search(r"episode_(?:[sf]_)?(\d+)", os.path.basename(p))
            eps.append({"path": p, "split": split,
                        "episode": int(m.group(1)) if m else -1,
                        "outcome": episode_outcome(p)})
    return eps


@lru_cache(maxsize=8)
def load_episode(path):
    """Load + stack one episode once; cached so the frame slider scrubs without re-reading 10 MB.

    Returns dict(pos (T,B,H,3) | None, exec (T,H,3) | None, rgb (T,h,w,3), meta).
    """
    with open(path, "rb") as fh:
        ep = pickle.load(fh)
    meta, roll = ep["metadata"], ep["rollout"]
    rgb = np.stack([to_displayable(s["rgb"]) for s in roll])              # (T, h, w, 3) uint8
    pos_idx = meta["action_mappings"].get("pos")
    if pos_idx is None:
        return dict(pos=None, exec=None, rgb=rgb, meta=meta)
    preds = np.stack([unwrap(s["action_pred"]) for s in roll])           # (T, B, H, A)
    acts = np.stack([unwrap(s["action"]) for s in roll])                 # (T, H, A) executed chunk
    return dict(pos=preds[..., pos_idx], exec=acts[..., pos_idx], rgb=rgb, meta=meta)


def chunks_figure(pos, exec_path, meta, stride=5, n_samples=None, colorscale="Viridis"):
    T = pos.shape[0]
    steps = list(range(0, T, stride))
    colors = sample_colorscale(colorscale, [t / max(T - 1, 1) for t in steps])

    fig = go.Figure()
    for t, color in zip(steps, colors):
        chunks = pos[t][:n_samples] if n_samples else pos[t]            # (B', H, 3)
        nan = np.full((chunks.shape[0], 1, 3), np.nan)                  # NaN breaks -> 1 trace/step
        xyz = np.concatenate([chunks, nan], axis=1).reshape(-1, 3)
        fig.add_trace(go.Scatter3d(
            x=xyz[:, 0], y=xyz[:, 1], z=xyz[:, 2], mode="lines",
            line=dict(color=color, width=2), opacity=0.4, showlegend=False))

    bb = exec_path[:, 0, :]                                             # executed route, 1 pt/step
    fig.add_trace(go.Scatter3d(
        x=bb[:, 0], y=bb[:, 1], z=bb[:, 2], mode="lines",
        line=dict(color="black", width=2, dash="dot"), opacity=0.45, name="executed"))

    fig.add_trace(go.Scatter3d(                                        # colorbar proxy for step
        x=[None], y=[None], z=[None], mode="markers",
        marker=dict(color=[0, T - 1], colorscale=colorscale, cmin=0, cmax=T - 1,
                    colorbar=dict(title="step"), showscale=True), showlegend=False))

    outcome = "SUCCESS" if meta["successful"] else "FAILURE"
    fig.update_layout(
        scene=dict(xaxis_title="x", yaxis_title="y", zaxis_title="z", aspectmode="data"),
        title=f"{meta['task']} · ep {meta['episode']} · {outcome}", autosize=True,
        margin=dict(l=0, r=0, t=40, b=0),
        uirevision="keep")   # preserve camera + active drag tool across figure rebuilds
    return fig


def rgb_animation(rgb, fps=12):
    """Native Plotly animation over all RGB frames: client-side play/pause + a step slider.

    Frames are embedded as compressed PNG strings (binary_string=True) rather than raw int
    arrays -- ~20x smaller payload, so the figure loads fast over a forwarded port.
    """
    w = rgb.shape[2]
    fig = px.imshow(rgb, animation_frame=0, binary_string=True,
                    labels=dict(animation_frame="step"))
    fig.add_vline(x=w // 2, line=dict(color="white", width=1, dash="dot"))  # divider: cam0 | cam1
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    fig.update_layout(margin=dict(l=0, r=0, t=36, b=0), autosize=True)
    # set playback speed (px.imshow's Play button is updatemenus[0].buttons[0])
    fig.layout.updatemenus[0].buttons[0].args[1]["frame"]["duration"] = 1000 // fps
    return fig


app = Dash(__name__)
app.layout = html.Div([
    html.Div([
        dcc.Dropdown(
            id="task", value="stacking", clearable=False, style={"width": "200px"},
            options=[{"label": t + ("  (velocity – N/A)" if t not in POSITION_TASKS else ""),
                      "value": t, "disabled": t not in POSITION_TASKS} for t in ALL_TASKS]),
        dcc.Dropdown(id="episode", clearable=False, style={"width": "340px"}),
        html.Button("🎲 success", id="rand-success", n_clicks=0),
        html.Button("🎲 fail", id="rand-fail", n_clicks=0),
        html.Label("stride"),
        html.Div(dcc.Slider(id="stride", min=1, max=20, step=1, value=5, marks=None,
                            tooltip={"placement": "bottom"}), style={"width": "240px"}),
    ], style={"display": "flex", "gap": "14px", "alignItems": "center", "padding": "8px"}),
    html.Div([
        # left:right = 2:1, basis 0 so they split purely by ratio; minWidth 0 lets both shrink
        dcc.Graph(id="fig", responsive=True,
                  style={"flex": "2 1 0", "minWidth": "0", "height": "100%"}),
        html.Div([
            # container, not a fixed Graph: we remount a fresh Graph per episode so a playing
            # animation is destroyed cleanly instead of glitching against the new figure
            html.Div(id="rgb-box", style={"flex": "1 1 0", "minHeight": "0", "display": "flex"}),
        ], style={"flex": "1 1 0", "minWidth": "0", "display": "flex",
                  "flexDirection": "column", "gap": "6px", "padding": "8px"}),
    ], style={"display": "flex", "alignItems": "stretch", "height": "82vh"}),
])


@app.callback(Output("episode", "options"), Output("episode", "value"), Input("task", "value"))
def _episodes(task):
    eps = list_episodes(task)
    opts = [{"label": f"{e['split']} · ep{e['episode']} · {e['outcome']}", "value": e["path"]}
            for e in eps]
    return opts, (opts[0]["value"] if opts else None)


@app.callback(
    Output("episode", "value", allow_duplicate=True),
    Input("rand-success", "n_clicks"), Input("rand-fail", "n_clicks"),
    State("task", "value"), prevent_initial_call=True)
def _random(_s, _f, task):
    want = "SUCCESS" if ctx.triggered_id == "rand-success" else "FAIL"
    pool = [e["path"] for e in list_episodes(task) if e["outcome"] == want]
    return random.choice(pool) if pool else no_update


@app.callback(Output("fig", "figure"), Input("episode", "value"), Input("stride", "value"))
def _chunks(path, stride):
    if not path:
        return go.Figure()
    d = load_episode(path)
    if d["pos"] is None:
        return go.Figure(layout=dict(title=f"{d['meta']['task']}: velocity action — no EE position"))
    return chunks_figure(d["pos"], d["exec"], d["meta"], stride=stride)


@app.callback(Output("rgb-box", "children"), Input("episode", "value"))
def _rgb(path):
    fig = rgb_animation(load_episode(path)["rgb"]) if path else go.Figure()
    # fresh Graph each call -> old (possibly playing) one unmounts, stopping its animation loop
    return dcc.Graph(figure=fig, responsive=True, style={"flex": "1 1 0", "minHeight": "0"})


if __name__ == "__main__":
    app.run(debug=True)
