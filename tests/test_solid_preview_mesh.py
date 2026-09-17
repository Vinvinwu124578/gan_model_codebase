"""Regression test for coherent solid STL geometry in route previews."""
from __future__ import annotations

from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import trimesh


TOOLS = Path(__file__).resolve().parents[1] / "tools"
sys.path.insert(0, str(TOOLS))
import auto_cr3_coverage_board_sampler as sampler  # noqa: E402


class SolidPreviewMeshTests(unittest.TestCase):
    def test_dense_stl_is_simplified_without_breaking_the_closed_surface(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "dense_box.stl"
            mesh = trimesh.creation.box(extents=(10.0, 20.0, 30.0))
            for _ in range(5):
                mesh = mesh.subdivide()
            mesh.export(path)

            vertices, faces = sampler.load_preview_mesh_arrays(path, face_limit=50)
            preview = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)

            self.assertLessEqual(len(faces), 50)
            self.assertTrue(preview.is_watertight)
            np.testing.assert_allclose(preview.extents, (10.0, 20.0, 30.0), atol=0.02)


if __name__ == "__main__":
    unittest.main(verbosity=2)
