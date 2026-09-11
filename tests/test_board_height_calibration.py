"""Offline synthetic checks; these observations are not robot calibration data."""
from __future__ import annotations

import copy
from contextlib import contextmanager
import json
from pathlib import Path
import sys
import unittest
import uuid

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from board_height_calibration import fit_height_calibration, height_correction_at, load_height_calibration, make_bindings

SYNTHETIC_BINDINGS = dict(fixture_profile_sha256="1" * 64, dock_design_sha256="2" * 64,
                          board_manifest_sha256="3" * 64, tile_id="tile_nw", user=0, tool=2)


@contextmanager
def workspace_test_directory():
    # Ordinary workspace ACLs avoid Windows sandbox trouble with mode 0700.
    path = Path(__file__).resolve().parents[1] / (".height-test-" + uuid.uuid4().hex)
    path.mkdir()
    try:
        yield path
    finally:
        for child in path.iterdir():
            child.unlink()
        path.rmdir()


def observations(bindings=None):
    rows = []
    for point, (role, x, y) in enumerate([
        ("fit", -50, -50), ("fit", 50, -50), ("fit", 50, 50), ("fit", -50, 50),
        ("validate", -20, 10), ("validate", 25, -5),
    ]):
        for repeat, noise in enumerate((-.002, 0, .002)):
            z = 20 + .001 * x - .002 * y - 2 + noise
            rows.append(dict(point_id=f"p{point}", role=role, repeat_id=str(repeat),
                             local_x_mm=x, local_y_mm=y, nominal_surface_z_mm=20,
                             contact_z_mm=z-.01, no_contact_z_mm=z+.01,
                             **(SYNTHETIC_BINDINGS if bindings is None else bindings)))
    return rows


class HeightCalibrationTests(unittest.TestCase):
    def fit(self, rows=None, **kwargs):
        params = dict(measurement_uncertainty_mm=.01, measurement_reference="SYNTHETIC UNIT TEST ONLY", bindings=SYNTHETIC_BINDINGS)
        params.update(kwargs)
        return fit_height_calibration(observations(params["bindings"]) if rows is None else rows, **params)

    def test_plane_sign_and_budget(self):
        p = self.fit()
        self.assertAlmostEqual(height_correction_at(p, 0, 0), -2)
        self.assertAlmostEqual(height_correction_at(p, 50, 50), -2.05)
        self.assertAlmostEqual(p["quality"]["additive_error_budget_mm"], .024)

    def test_default_point_one_mm_target_rejects_accumulated_error(self):
        accepted = self.fit(measurement_uncertainty_mm=.08)
        self.assertAlmostEqual(accepted["limits"]["max_error_budget_mm"], .10)
        self.assertAlmostEqual(accepted["quality"]["additive_error_budget_mm"], .094)
        # All component gates pass, but .09 + .01 + .004 > the .10 mm target.
        with self.assertRaisesRegex(ValueError, "budget .* exceeds limit 0.100000"):
            self.fit(measurement_uncertainty_mm=.09)

    def test_strict_default_component_limits(self):
        rows = observations()
        for row in rows:
            row["no_contact_z_mm"] = row["contact_z_mm"] + .05
        with self.assertRaisesRegex(ValueError, "bracket"):
            self.fit(rows)
        rows = observations()
        for key in ("contact_z_mm", "no_contact_z_mm"):
            rows[0][key] += .04
        with self.assertRaisesRegex(ValueError, "repeat spread"):
            self.fit(rows)
        rows = observations()
        for row in rows:
            if row["point_id"] == "p4":
                row["contact_z_mm"] += .04
                row["no_contact_z_mm"] += .04
        with self.assertRaisesRegex(ValueError, "validation residual"):
            self.fit(rows)

    def test_no_extrapolation(self):
        with self.assertRaisesRegex(ValueError, "extrapolation"):
            height_correction_at(self.fit(), 50.001, 0)

    def test_non_finite_input_rejected(self):
        for name in ("local_x_mm", "contact_z_mm", "no_contact_z_mm"):
            rows = observations()
            rows[0][name] = float("nan")
            with self.subTest(name=name), self.assertRaises(ValueError):
                self.fit(rows)

    def test_reversed_and_wide_brackets(self):
        for width in (-.1, 0, .3):
            rows = observations()
            rows[0]["no_contact_z_mm"] = rows[0]["contact_z_mm"] + width
            with self.subTest(width=width), self.assertRaisesRegex(ValueError, "bracket"):
                self.fit(rows)

    def test_repeats_required_and_unique(self):
        rows = observations()
        with self.assertRaisesRegex(ValueError, "three"):
            self.fit(rows[1:])
        rows[1]["repeat_id"] = rows[0]["repeat_id"]
        with self.assertRaisesRegex(ValueError, "uniquely"):
            self.fit(rows)

    def test_repeat_geometry_consistency(self):
        rows = observations()
        rows[0]["nominal_surface_z_mm"] += 1
        with self.assertRaisesRegex(ValueError, "inconsistent"):
            self.fit(rows)

    def test_repeat_spread(self):
        rows = observations()
        for key in ("contact_z_mm", "no_contact_z_mm"):
            rows[0][key] += .3
        with self.assertRaisesRegex(ValueError, "repeat spread"):
            self.fit(rows)

    def test_validation_error_and_budget(self):
        rows = observations()
        for row in rows:
            if row["point_id"] == "p4":
                row["contact_z_mm"] += .4
                row["no_contact_z_mm"] += .4
        with self.assertRaisesRegex(ValueError, "validation residual"):
            self.fit(rows)
        with self.assertRaisesRegex(ValueError, "budget"):
            self.fit(measurement_uncertainty_mm=.8)

    def test_independent_reference_required(self):
        for value in (0, -.1, float("inf"), None):
            with self.subTest(value=value), self.assertRaises(ValueError):
                self.fit(measurement_uncertainty_mm=value)
        with self.assertRaises(ValueError):
            self.fit(measurement_reference="")

    def test_validation_independence_and_support(self):
        rows = observations()
        for row in rows:
            if row["point_id"] == "p4":
                row["local_x_mm"], row["local_y_mm"] = -50, -50
        with self.assertRaisesRegex(ValueError, "independent"):
            self.fit(rows)
        rows = observations()
        for row in rows:
            if row["point_id"] == "p4":
                row["local_x_mm"] = 60
        with self.assertRaisesRegex(ValueError, "outside"):
            self.fit(rows)

    def test_collinear_fit(self):
        rows = observations()
        for row in rows:
            if row["role"] == "fit":
                row["local_x_mm"] = int(row["point_id"][1:]) * 10
                row["local_y_mm"] = 0
        with self.assertRaisesRegex(ValueError, "non-collinear"):
            self.fit(rows)

    def test_ill_conditioned_fit(self):
        rows = observations()
        for row in rows:
            if row["role"] == "fit":
                row["local_y_mm"] *= .00001
        with self.assertRaisesRegex(ValueError, "ill-conditioned"):
            self.fit(rows)

    def test_magnitude_and_slope(self):
        with self.assertRaisesRegex(ValueError, "magnitude"):
            self.fit(limits={"max_correction_mm": 1})
        with self.assertRaisesRegex(ValueError, "slope"):
            self.fit(limits={"max_slope_deg": .01})

    def test_validation_count(self):
        rows = [row for row in observations() if row["point_id"] != "p5"]
        with self.assertRaisesRegex(ValueError, "two independent"):
            self.fit(rows)

    def test_each_measurement_must_match_bindings(self):
        rows = observations()
        rows[0]["fixture_profile_sha256"] = "f" * 64
        with self.assertRaisesRegex(ValueError, "bindings differ"):
            self.fit(rows)
        rows = observations()
        del rows[0]["tool"]
        with self.assertRaisesRegex(ValueError, "lacks fields"):
            self.fit(rows)

    def test_load_binding_and_model_tamper(self):
        with workspace_test_directory() as root:
            fixture, dock, manifest = [root / name for name in ("fixture.json", "dock.json", "manifest.json")]
            for path in (fixture, dock, manifest):
                path.write_text("{}", encoding="utf-8")
            bindings = make_bindings(fixture, dock, manifest, "tile_nw", 0, 2)
            profile = self.fit(bindings=bindings)
            saved = root / "height.json"
            saved.write_text(json.dumps(profile), encoding="utf-8")
            loaded = load_height_calibration(saved, fixture, dock, manifest, "tile_nw", 0, 2)
            self.assertAlmostEqual(height_correction_at(loaded, 0, 0), -2)
            with self.assertRaisesRegex(ValueError, "another"):
                load_height_calibration(saved, fixture, dock, manifest, "tile_ne", 0, 2)
            damaged = copy.deepcopy(profile)
            damaged["coefficients_mm"][2] += 1
            saved.write_text(json.dumps(damaged), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "measurement audit"):
                load_height_calibration(saved, fixture, dock, manifest, "tile_nw", 0, 2)
            saved.write_text(json.dumps(profile), encoding="utf-8")
            fixture.write_text('{"changed": true}', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "another"):
                load_height_calibration(saved, fixture, dock, manifest, "tile_nw", 0, 2)


if __name__ == "__main__":
    unittest.main()
