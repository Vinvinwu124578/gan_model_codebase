"""Verify display-only mesh reuse offline without importing robot or mesh drivers."""

import ast
from functools import lru_cache
import os
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch
from uuid import uuid4

import numpy as np


SOURCE = Path(__file__).resolve().parents[1] / "tools" / "auto_cr3_coverage_board_sampler.py"
tree = ast.parse(SOURCE.read_text(encoding="utf-8-sig"), filename=str(SOURCE))
selected = [ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)]
selected.extend(node for node in tree.body if isinstance(node, ast.FunctionDef)
                and node.name in {
                    "load_preview_mesh_arrays",
                    "_cached_preview_mesh_arrays",
                    "preview_height_intensity",
                })
ns = {"Path": Path, "np": np, "lru_cache": lru_cache}
exec(compile(ast.fix_missing_locations(ast.Module(body=selected, type_ignores=[])), str(SOURCE), "exec"), ns)


class Mesh:
    def __init__(self):
        self.vertices = np.arange(90, dtype=float).reshape(30, 3)
        self.faces = np.arange(30, dtype=int).reshape(10, 3)


class PreviewCacheTests(unittest.TestCase):
    def setUp(self):
        ns["_cached_preview_mesh_arrays"].cache_clear()
        # Ordinary mkdir inherits workspace permissions on Windows; tempfile's
        # explicit 0700 ACL prevents access under the restricted test token.
        self.folder = SOURCE.parent / ("cr3_preview_cache_" + uuid4().hex)
        self.folder.mkdir()
        self.assertTrue(self.folder.resolve().is_relative_to(SOURCE.parent.resolve()))
        self.addCleanup(self.cleanup_folder)
        self.path = self.folder / "tile.stl"
        self.path.write_bytes(b"mock geometry")
        self.mesh = Mesh()
        self.loader = Mock(return_value=self.mesh)
        stub = SimpleNamespace(Trimesh=Mesh, load_mesh=self.loader)
        self.modules = patch.dict(sys.modules, {"trimesh": stub})
        self.modules.start()
        self.addCleanup(self.modules.stop)

    def cleanup_folder(self):
        for child in self.folder.iterdir():
            if child.is_dir():
                child.rmdir()
            else:
                child.unlink()
        self.folder.rmdir()

    def test_three_previews_share_read_only_arrays(self):
        a = ns["load_preview_mesh_arrays"](self.path, 4)
        b = ns["load_preview_mesh_arrays"](self.path, 4)
        c = ns["load_preview_mesh_arrays"](self.path, 4)
        self.assertIs(a, b)
        self.assertIs(a, c)
        self.assertEqual(self.loader.call_count, 1)
        vertices, faces = a
        # A simplifier failure must preserve the complete connected mesh.
        # Randomly selecting four faces would recreate the old foggy preview.
        self.assertEqual(faces.shape, (10, 3))
        self.assertTrue(np.all(faces >= 0))
        self.assertTrue(np.all(faces < len(vertices)))
        for array in a:
            self.assertFalse(array.flags.writeable)
            with self.assertRaises(ValueError):
                array.flat[0] = 123
        self.assertFalse(np.shares_memory(vertices, self.mesh.vertices))

    def test_file_size_mtime_and_face_limit_invalidate(self):
        ns["load_preview_mesh_arrays"](self.path, 4)
        self.path.write_bytes(b"changed mock geometry")
        ns["load_preview_mesh_arrays"](self.path, 4)
        timestamp = self.path.stat().st_mtime_ns + 1_000_000_000
        os.utime(self.path, ns=(timestamp, timestamp))
        ns["load_preview_mesh_arrays"](self.path, 4)
        vertices, faces = ns["load_preview_mesh_arrays"](self.path, 12)
        self.assertEqual(self.loader.call_count, 4)
        self.assertEqual(faces.shape, (10, 3))
        np.testing.assert_array_equal(vertices, self.mesh.vertices)

    def test_cache_is_bounded_and_paths_are_normalized(self):
        ns["load_preview_mesh_arrays"](self.path, 4)
        (self.path.parent / "sub").mkdir()
        ns["load_preview_mesh_arrays"](self.path.parent / "sub" / ".." / "tile.stl", 4)
        self.assertEqual(self.loader.call_count, 1)
        for index in range(5):
            path = self.path.with_name("other_{}.stl".format(index))
            path.write_bytes(b"mock")
            ns["load_preview_mesh_arrays"](path, 4)
        self.assertEqual(ns["_cached_preview_mesh_arrays"].cache_info().currsize, 4)

    def test_preview_height_intensity_is_robust_and_bounded(self):
        vertices = np.column_stack(
            (
                np.zeros(101),
                np.zeros(101),
                np.concatenate((np.arange(100, dtype=float), np.asarray((10000.0,)))),
            )
        )
        intensity = ns["preview_height_intensity"](vertices)
        self.assertEqual(intensity.shape, (101,))
        self.assertTrue(np.all(intensity >= 0.0))
        self.assertTrue(np.all(intensity <= 1.0))
        self.assertEqual(float(intensity[-1]), 1.0)
        self.assertGreater(float(intensity[75]), float(intensity[25]))

    def test_preview_height_intensity_handles_flat_mesh(self):
        vertices = np.asarray(((0.0, 0.0, 4.0), (1.0, 1.0, 4.0)))
        np.testing.assert_allclose(ns["preview_height_intensity"](vertices), 0.5)


if __name__ == "__main__":
    unittest.main(verbosity=2)
