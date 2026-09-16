# AgentSysID Data Contract

## Required columns

| Column | Type | Description |
|--------|------|-------------|
| `time` | float | Monotonic or piecewise-monotonic timestamp in seconds |
| `s_*`  | float | State variables (one column per state) |
| `a_*`  | float | Action / input variables (one column per input) |

## Optional columns

| Column | Type | Description |
|--------|------|-------------|
| `xdot_*` | float | True state derivatives. When present and matching the number of `s_*` columns, they are used as training targets instead of finite-difference estimates. |

## Multi-trajectory files

- Set `MULTI_TRAJECTORY = True` (or rely on auto-detection via large state jumps / time resets).
- Trajectory boundaries are detected when any state jump exceeds `RESET_THRESHOLD` or when time goes backwards.

## Encoding conventions

- Angle states that should be wrapped to \([-\pi, \pi]\) can be listed in `ANGLE_INDICES` (0-based indices into the `s_*` column order).
- Column order of `s_*` / `a_*` defines the dimension ordering used by the network.
