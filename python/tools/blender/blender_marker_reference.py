# SPDX-FileCopyrightText: 2026 Harri Kaimio
#
# SPDX-License-Identifier: Apache-2.0

"""
blender_marker_reference.py — place marker slots on a reference body in Blender.

Markers are placed once on a reference model (a Character Creator export, say), snapped
to its surface, and the placements are exchanged with posetrak as a small JSON file
(posetrak/markers/reference.py describes it and the frame it is in). The frame needs only
bone positions, so any armature with posetrak's joint names works: the one built from the
file, or a template armature already aligned to the character.

Commands (run inside Blender; arguments follow a ``--``)::

    blender --python blender_marker_reference.py -- build   reference.json [--armature NAME] [--frame N]
    blender --python blender_marker_reference.py -- snap    MESH_OBJECT[,MESH_OBJECT...]
    blender --python blender_marker_reference.py -- export  placements.json [--armature NAME]

build
    Makes an armature from the document's bones if the scene has none named *NAME* (default
    ``PosetrakReference``; in a scene with an armature already posed onto the model, pass its name),
    and one marker per slot: a sphere empty at the marker with an arrow
    along its normal, both under the empty ``mk_<slot>``.
snap
    Moves every marker to the nearest point of the mesh surface and turns its arrow along the
    surface normal (outwards). Then adjust by hand: grab an empty and move it over the surface
    (surface snapping on), and re-run ``snap`` to refresh the normals.
export
    Writes each marker's placement on its parent bone: ``along``, ``lateral``, ``anterior``
    and ``normal``, ready for ``posetrak.markers.reference``.

In an interactive session call the functions from Blender's Python console::

    import sys; sys.path.insert(0, r"D:/mocap/posetrak/python/tools/blender")
    import blender_marker_reference as bmr
    bmr.snap_to_surface("CC_Body,Boxers")
    bmr.export("placements.json", "NAME-OF-ARMATURE")

The bones' positions are taken from the armature's current pose, so a BVH of a tracking run posed
onto the model works as the armature.
"""
import json
import sys

import bpy
import mathutils

DEFAULT_ARMATURE = "PosetrakReference"
# Bones the body axes are made from. Rename if the armature calls them something else.
AXIS_BONES = {"right": "shoulder.R", "left": "shoulder.L", "hips": "hips", "head": "head"}
FORWARD_BONE_LIMIT = 0.3
MARKER_PREFIX = "mk_"


# --------------------------------------------------------------------------------------
# frames (the same maths as posetrak/markers/reference.py)
# --------------------------------------------------------------------------------------

def bone_ends(arm, name):
    """World head and tail of a bone in the armature's current pose (its rest pose if it has none).

    The pose is what matches the model when the armature was posed or animated onto it (a BVH
    of a tracking run, say), and the rest pose of such an armature need not be a standing one.
    """
    bone = arm.pose.bones[name]
    m = arm.matrix_world
    return m @ bone.head, m @ bone.tail


def body_axes(arm):
    """(right, front, up) unit vectors of the body."""
    names = AXIS_BONES
    right = bone_ends(arm, names["right"])[0] - bone_ends(arm, names["left"])[0]
    right.normalize()
    up = bone_ends(arm, names["head"])[0] - bone_ends(arm, names["hips"])[0]
    up = up - up.dot(right) * right
    up.normalize()
    return right, up.cross(right), up


def bone_frame(arm, name, axes):
    """(head, length, long, lateral, anterior) of a bone: its direction, and the directions
    across it (see posetrak/markers/reference.py)."""
    _, front, up = axes
    head, tail = bone_ends(arm, name)
    d = tail - head
    length = d.length
    d.normalize()
    anterior = front - front.dot(d) * d
    if anterior.length < FORWARD_BONE_LIMIT:
        anterior = up - up.dot(d) * d
    anterior.normalize()
    lateral = d.cross(anterior)
    return head, length, d, lateral, anterior


# --------------------------------------------------------------------------------------
# build
# --------------------------------------------------------------------------------------

def build_armature(bones, name=DEFAULT_ARMATURE):
    """An armature of the given rest-pose bones ({name: {head, tail, parent}})."""
    data = bpy.data.armatures.new(name)
    obj = bpy.data.objects.new(name, data)
    bpy.context.scene.collection.objects.link(obj)
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.mode_set(mode="EDIT")
    made = {}
    for bone_name, ends in bones.items():
        bone = data.edit_bones.new(bone_name)
        bone.head = mathutils.Vector(ends["head"])
        bone.tail = mathutils.Vector(ends["tail"])
        made[bone_name] = bone
    for bone_name, ends in bones.items():
        if ends.get("parent") in made:
            made[bone_name].parent = made[ends["parent"]]
    bpy.ops.object.mode_set(mode="OBJECT")
    obj.show_in_front = True
    return obj


def make_marker(name, location, normal, parent_joint, confidence=None, size=0.012):
    """An empty ``mk_<name>`` whose +Z arrow points along *normal*, with a sphere child at the marker."""
    arrow = bpy.data.objects.new(MARKER_PREFIX + name, None)
    arrow.empty_display_type = "SINGLE_ARROW"
    arrow.empty_display_size = 0.05
    arrow.location = location
    z = mathutils.Vector(normal) if normal is not None else mathutils.Vector((0, 0, 1))
    arrow.rotation_mode = "QUATERNION"
    arrow.rotation_quaternion = z.to_track_quat("Z", "Y")
    arrow["parent_joint"] = parent_joint
    if confidence:
        arrow["confidence"] = confidence
    bpy.context.scene.collection.objects.link(arrow)
    dot = bpy.data.objects.new(MARKER_PREFIX + name + ".dot", None)
    dot.empty_display_type = "SPHERE"
    dot.empty_display_size = size
    dot.parent = arrow
    bpy.context.scene.collection.objects.link(dot)
    return arrow


def find_armature(name):
    obj = bpy.data.objects.get(name)
    if obj is not None and obj.type == "ARMATURE":
        return obj
    return None


def build(document_path, armature_name=DEFAULT_ARMATURE):
    doc = json.load(open(document_path, encoding="utf-8"))
    arm = find_armature(armature_name)
    if arm is None:
        if not doc.get("bones"):
            raise SystemExit("no armature named %r in the scene and the document has no bones" % armature_name)
        arm = build_armature(doc["bones"], armature_name)
    axes = body_axes(arm)
    for m in doc["markers"]:
        head, length, d, lateral, anterior = bone_frame(arm, m["parent_joint"], axes)
        location = head + d * (m["along"] * length) + lateral * m["lateral"] + anterior * m["anterior"]
        normal = None
        if m.get("normal"):
            n = m["normal"]
            normal = d * n[0] + lateral * n[1] + anterior * n[2]
        make_marker(m["name"], location, normal, m["parent_joint"], m.get("confidence"))
    return arm


# --------------------------------------------------------------------------------------
# snap and export
# --------------------------------------------------------------------------------------

def markers():
    bpy.context.view_layer.update()      # world matrices of objects made through the data API are stale until then
    return [o for o in bpy.data.objects if o.name.startswith(MARKER_PREFIX) and not o.name.endswith(".dot")]


def nearest_surface(mesh_names, location):
    """(distance, point, outward normal) of the surface point nearest *location* over the meshes."""
    graph = bpy.context.evaluated_depsgraph_get()
    best = None
    for name in mesh_names:
        mesh = bpy.data.objects[name]
        to_local = mesh.matrix_world.inverted()
        ok, point, normal, _ = mesh.closest_point_on_mesh(to_local @ location, depsgraph=graph)
        if not ok:
            continue
        world_point = mesh.matrix_world @ point
        distance = (world_point - location).length
        if best is None or distance < best[0]:
            best = (distance, world_point, (mesh.matrix_world.to_3x3() @ normal).normalized())
    return best


def snap_to_surface(mesh_names):
    """Move each marker to the nearest surface point of the meshes (a name or a list of names) and
    point its arrow outwards along the normal. Returns each marker's distance moved."""
    bpy.context.view_layer.update()
    if isinstance(mesh_names, str):
        mesh_names = [n for n in mesh_names.split(",") if n]
    moved = {}
    for m in markers():
        hit = nearest_surface(mesh_names, m.matrix_world.translation)
        if hit is None:
            continue
        moved[m.name[len(MARKER_PREFIX):]] = hit[0]
        m.location = hit[1]
        m.rotation_mode = "QUATERNION"
        m.rotation_quaternion = hit[2].to_track_quat("Z", "Y")
    return moved


def export(path, armature_name=DEFAULT_ARMATURE):
    arm = find_armature(armature_name)
    if arm is None:
        raise SystemExit("no armature named %r" % armature_name)
    axes = body_axes(arm)
    out = []
    for m in markers():
        joint = m["parent_joint"]
        head, length, d, lateral, anterior = bone_frame(arm, joint, axes)
        p = m.matrix_world.translation - head
        n = m.matrix_world.to_3x3() @ mathutils.Vector((0, 0, 1))
        out.append({
            "name": m.name[len(MARKER_PREFIX):],
            "parent_joint": joint,
            "along": p.dot(d) / length,
            "lateral": p.dot(lateral),
            "anterior": p.dot(anterior),
            "normal": [n.dot(d), n.dot(lateral), n.dot(anterior)],
        })
    json.dump({"markers": sorted(out, key=lambda r: r["name"])}, open(path, "w", encoding="utf-8"), indent=1)
    return out


def main(argv):
    args = argv[argv.index("--") + 1:] if "--" in argv else []
    if not args:
        print(__doc__)           # run from Blender's text editor: the functions above are for the console
        return
    command, rest = args[0], args[1:]
    name = DEFAULT_ARMATURE
    if "--armature" in rest:
        i = rest.index("--armature")
        name = rest[i + 1]
        rest = rest[:i] + rest[i + 2:]
    if "--frame" in rest:
        i = rest.index("--frame")
        bpy.context.scene.frame_set(int(rest[i + 1]))
        rest = rest[:i] + rest[i + 2:]
    if command == "build":
        build(rest[0], name)
    elif command == "snap":
        snap_to_surface(rest[0])
    elif command == "export":
        export(rest[0], name)
    else:
        raise SystemExit("unknown command %r" % command)


if __name__ == "__main__":
    main(sys.argv)
