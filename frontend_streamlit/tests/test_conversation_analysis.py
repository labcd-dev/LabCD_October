import json
import numpy as np
import pandas as pd

from frontend_streamlit.conversation_analysis import compare_run_states, profile_frame


def test_wrap_aware_derivative_check_unwraps_angle_before_differencing():
    time = np.arange(101, dtype=float) * 0.01
    physical_angle = np.linspace(2.8, 3.5, len(time))
    stored_angle = (physical_angle + np.pi) % (2 * np.pi) - np.pi
    frame = pd.DataFrame({
        "time": time,
        "s_pitch": stored_angle,
        "xdot_pitch": np.gradient(physical_angle, time),
    })

    result = profile_frame(frame, ["s_pitch"], [], ["xdot_pitch"])

    assert result["detected_wrap_states"] == ["s_pitch"]
    check = result["derivative_consistency"][0]
    assert check["angle_unwrapped_for_check"] is True
    assert check["finite_difference_vs_supplied_rmse"] < 1e-8
    assert check["rmse_over_supplied_std"] is None


def test_nonwrapped_derivative_check_uses_original_signal():
    time = np.arange(20, dtype=float) * 0.05
    position = time**2
    frame = pd.DataFrame({"time": time, "s_position": position,
                          "xdot_position": 2 * time})

    result = profile_frame(frame, ["s_position"], [], ["xdot_position"])

    check = result["derivative_consistency"][0]
    assert check["angle_unwrapped_for_check"] is False
    assert check["rmse_over_supplied_std"] is not None


def test_state_comparison_keeps_per_trajectory_breakdown(tmp_path):
    (tmp_path / "verification_summary.json").write_text(json.dumps({
        "aligned_samples": 30,
        "states": [{"state": "s_pitch", "rmse": .2, "mae": .1,
                    "rmse_over_test_std": .3, "valid_samples": 30,
                    "error_type": "circular"}],
        "per_trajectory": [
            {"trajectory_index": 1, "states": [{"state": "s_pitch", "rmse": .1,
                "mae": .05, "rmse_over_test_std": .15, "valid_samples": 12,
                "error_type": "circular"}]},
            {"trajectory_index": 2, "states": [{"state": "s_pitch", "rmse": .3,
                "mae": .15, "rmse_over_test_std": .45, "valid_samples": 18,
                "error_type": "circular"}]},
        ],
    }), encoding="utf-8")

    comparison = compare_run_states(str(tmp_path))

    assert [item["trajectory_index"] for item in comparison["rows"][0]["per_trajectory"]] == [1, 2]
    assert comparison["rows"][0]["per_trajectory"][1]["normalized_rmse"] == .45
    assert comparison["rows"][0]["error_type"] == "circular"
