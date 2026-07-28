# Data

Robot-policy rollouts for five manipulation tasks, used for failure prediction / calibration. Each rollout is a single episode of a policy executing a task, stored as a pickled Python dict.

## Layout

```
data/<task>/rollouts/
├── calibration/         # successful rollouts used to calibrate detectors
├── calibration_unused/  # extra calibration episodes not used (stacking, sorting only)
├── test/                # held-out rollouts (mix of success/failure) for evaluation
├── videos/              # optional .mp4 renders mirroring calibration/ and test/ (stacking, sorting)
└── rollout_statistics.jpg  # summary plot (push_t, push_chair)
```

Tasks: `stacking`, `sorting`, `push_t`, `pretzel`, `push_chair`.

| Task        | calibration | test | videos |
|-------------|------------:|-----:|-------:|
| stacking    | 50          | 800  | yes    |
| sorting     | 50          | 400  | yes    |
| push_t      | 50          | 300  | no     |
| pretzel     | 10          | 20   | no     |
| push_chair  | 10          | 20   | no     |

## File naming

- `episode_s_XXXX.pkl` — success; `episode_f_XXXX.pkl` — failure. Used by **stacking, sorting, push_t, and push_chair**.
- `episode_N.pkl` — **pretzel only** (success flag lives in metadata).
- Calibration episodes are all successes (`_s_`). Test sets mix both.

## Pickle format

Each `.pkl` is a dict with two top-level keys, `metadata` and `rollout`.

### `metadata` (dict)
- `task`, `episode`, `num_robots`, `num_steps`
- `successful` (bool) — ground-truth outcome
- `rollout_type` — `calibration` or `test`
- `rollout_subtype` — finer split, task-dependent (see below); `None` for pretzel/push_chair
- `action_prediction_horizon`, `action_execution_horizon`, `action_batch_size`
- `action_mappings` — maps action-vector slices to `pos`/`vel`/`gripper`/`rpy`/`drpy`/`joint_pos`/`joint_vel` (key absent or `None` where unused); see [Action mappings](#action-mappings)
- feature-presence flags, **named differently per task**: `has_encoder_feat` / `has_state_feat` (stacking, sorting), `has_obs_embedding` / `has_state_embedding` (push_t, push_chair), or absent entirely (pretzel)
- there is also a redundant `metadata: True` entry inside the dict

`rollout_subtype` values by task:

| Task        | subtypes |
|-------------|----------|
| stacking    | `id` (in-dist), `eb`, `rb`, `mt` shifts |
| sorting      | `id`, `eb`, `sb`, `mt` shifts |
| push_t      | `hh`, `na` |
| pretzel, push_chair | none (`None`) |

### `rollout` (list, length `num_steps`)
One dict per timestep:

| key             | shape                  | notes |
|-----------------|------------------------|-------|
| `rgb`           | task-dependent, see below | rendered observation |
| `action`        | `(pred_horizon, A)`    | the predicted action chunk (length = `action_prediction_horizon`, **not** the execution horizon). For push_t/push_chair this is wrapped in a length-1 list (one ndarray per robot). |
| `action_pred`   | `(batch, pred_horizon, A)` | sampled action predictions (the policy's action-batch distribution); same list-wrapping for push_t/push_chair |
| `agent_pos`     | `(D,)` float32 or None | proprioceptive state (None for push_t) |
| `obs_embedding` | `(E,)` float32         | encoder feature |
| `state_embedding` | `(S,)` float32, None, or absent | state feature (populated only for push_t/push_chair) |
| `timestamp`, `step` | scalars            | |

The per-step `action_pred` batch is the key signal: the spread/uncertainty across sampled action predictions feeds the failure-prediction methods.

### Per-task shapes

| Task        | A  | pred_h | exec_h | batch | rgb            | obs_emb (E) | state_emb (S) | agent_pos (D) |
|-------------|---:|-------:|-------:|------:|----------------|------------:|--------------:|--------------:|
| stacking    | 21 | 8      | 4      | 32    | `96×192×3`     | 128         | —             | 8             |
| sorting      | 3  | 8      | 4      | 32    | `96×192×3`     | 128         | —             | 3             |
| push_t      | 3  | 16     | 8      | 256   | `512×512×3`    | 64          | 146           | — (None)      |
| pretzel     | 5  | 16     | 8      | 30    | `3×240×320` (CHW) | 512      | —             | 5             |
| push_chair  | 3  | 16     | 4      | 256   | `270×480×3`    | 96          | 216           | 13            |

Notes: `rgb` is `(H,W,3)` uint8 except **pretzel**, which is channels-first `(3,H,W)`. `action`/`action_pred` are float64 for stacking/sorting, float32 elsewhere.

### Action mappings

`action_mappings` says which slice of the A-dim action vector means what. It differs by task:

- **stacking** (A=21): `pos` `[0,1,2]`, `rpy` `[3,4,5]`, `joint_pos` `[6..12]`, `joint_vel` `[13..19]`, `gripper` `[20]`.
- **sorting** (A=3): `vel` `[0,1,2]` (velocity control; no position or gripper).
- **push_t**, **pretzel**, **push_chair**: `pos` `[0,1,2]` (push_t/push_chair also list `rpy`/`drpy`/`gripper` as `None`).

**Grasp/gripper is a continuous action, present only in stacking.** It is index 20 of stacking's action vector (`gripper: [20]`) — a continuous gripper command (observed range ~0.044–0.085, ~90 distinct values across an episode), not a binary open/close. The other four tasks have no gripper dimension. There is no separate named grasp *observation* variable; proprioception is the unlabeled `agent_pos` vector plus the encoder/state embeddings.
