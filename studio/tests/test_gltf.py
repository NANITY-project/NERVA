import json, math, os, shutil, struct, subprocess, sys, tempfile, unittest
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import gltf_import as g


def glb(doc):
    j = json.dumps(doc).encode(); j += b" " * (-len(j) % 4)
    return struct.pack("<4sII", b"glTF", 2, 12 + 8 + len(j)) + struct.pack("<II", len(j), 0x4E4F534A) + j


def box_doc(extra_node=None, **node):
    nodes = [dict({"mesh": 0, "name": "Crate"}, **node)] + ([extra_node] if extra_node else [])
    return {"asset": {"version": "2.0"}, "scene": 0, "scenes": [{"nodes": list(range(len(nodes)))}], "nodes": nodes,
            "meshes": [{"primitives": [{"attributes": {"POSITION": 0}}]}],
            "accessors": [{"min": [-1, -0.5, -0.5], "max": [1, 0.5, 0.5]}]}


class Hand(unittest.TestCase):
    def test_translation_and_scale(self):
        (p,), _, b = g.props_from_gltf(box_doc(translation=[3, 1, -2], scale=[2, 1, 1]))
        self.assertEqual([round(v, 6) for v in p["pos"]], [3, 1, -2]); self.assertEqual([round(v, 6) for v in p["size"]], [4, 1, 1])

    def test_rotation_swaps_extents(self):
        s = math.sin(math.pi / 4)
        (p,), _, _ = g.props_from_gltf(box_doc(rotation=[0, s, 0, s]))           # 90 deg about Y: x extent <-> z extent
        self.assertAlmostEqual(p["size"][0], 1.0, places=5); self.assertAlmostEqual(p["size"][2], 2.0, places=5)

    def test_hierarchy_and_glb_container(self):
        doc = box_doc(); doc["nodes"][0] = {"name": "Parent", "translation": [10, 0, 0], "children": [1]}
        doc["nodes"].append({"name": "Child", "mesh": 0, "translation": [0, 2, 0]}); doc["scenes"][0]["nodes"] = [0]
        (p,), _, _ = g.props_from_gltf(g.parse(glb(doc)))
        self.assertEqual([round(v, 6) for v in p["pos"]], [10, 2, 0])

    def test_filters_tags_and_skips(self):
        doc = box_doc(name="Dining_Table")
        doc["nodes"].append({"name": "Floor", "mesh": 0}); doc["scenes"][0]["nodes"] = [0, 1]
        props, skipped, _ = g.props_from_gltf(doc)
        self.assertEqual([p["name"] for p in props], ["Dining_Table"]); self.assertIn("fixed", props[0]["tags"]); self.assertIn("surface", props[0]["tags"])
        self.assertEqual(skipped[0][0], "Floor")

    def test_bad_input_raises_cleanly(self):
        for bad in (b"", b"hello world, definitely not gltf", b"glTF\x02\x00\x00\x00\x10\x00\x00\x00", json.dumps({"nodes": [{"matrix": [1, 2]}], "scenes": [{"nodes": [0]}]}).encode()):
            with self.assertRaises(g.GltfError, msg=bad[:20]): g.props_from_gltf(g.parse(bad))
        self.assertEqual(g.props_from_gltf({"nodes": [{"mesh": 0}], "meshes": [{"primitives": [{"attributes": {}}]}]})[1][0][1], "no POSITION bounds")


@unittest.skipUnless(shutil.which("blender"), "blender not installed")
class RealBlenderExport(unittest.TestCase):
    def test_matches_blender_ground_truth(self):
        d = tempfile.mkdtemp(); out = os.path.join(d, "kitchen")
        r = subprocess.run(["blender", "-b", "--factory-startup", "--python", os.path.join(os.path.dirname(__file__), "make_glb.py"), "--", out],
                           capture_output=True, text=True, timeout=120)
        self.assertIn("EXPORTED", r.stdout, r.stdout[-500:] + r.stderr[-500:])
        with open(out + ".truth.json") as f:
            truth = json.load(f)
        with open(out + ".glb", "rb") as f:
            props, skipped, bounds = g.import_bytes(f.read())
        names = {p["name"] for p in props}
        self.assertEqual(names, {"Table", "Cup", "Chair", "Crate", "Lantern"}, f"{names} / skipped {skipped}")
        self.assertEqual({s[0] for s in skipped}, {"Floor", "Wall_North"})
        worst = 0.0
        for p in props:
            t = truth[p["name"]]
            lo = [p["pos"][k] - p["size"][k] / 2 for k in range(3)]; hi = [p["pos"][k] + p["size"][k] / 2 for k in range(3)]
            e = max(max(abs(lo[k] - t["min"][k]), abs(hi[k] - t["max"][k])) for k in range(3))
            worst = max(worst, e)
            self.assertLess(e, 0.01, f"{p['name']}: box off by {e*1000:.1f} mm")
        print(f"\n  real Blender GLB: {len(props)} props, worst AABB error vs Blender's own geometry {worst*1000:.2f} mm")
        lantern = next(p for p in props if p["name"] == "Lantern")
        self.assertEqual(sorted(lantern["tags"]), ["fragile", "light"], "Blender custom property -> glTF extras -> tags")
        chair = next(p for p in props if p["name"] == "Chair")
        self.assertGreater(chair["size"][0], 0.46, "rotated chair AABB must be wider than its 0.45 m local size")
        crate = next(p for p in props if p["name"] == "Crate")
        self.assertGreater(abs(crate["pos"][0]) + abs(crate["pos"][2]), 0.5, "child-of-table transform must apply")
        self.assertIsNotNone(bounds)


if __name__ == "__main__":
    unittest.main(verbosity=2)
