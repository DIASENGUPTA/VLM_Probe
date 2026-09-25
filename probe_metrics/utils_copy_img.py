import os
import sys
import re
import json
import random
import subprocess
from typing import Tuple, List, Optional

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import networkx as nx

import plot_utils as plot_utils

def generate_random_seed():
    ran_seed = random.randint(0, 10000000)
    return ran_seed

BULK_EXAMPLES = ["sofa","chair","table","bed","wardrobe","cabinet","desk","bookshelf","counter","bench","tree", "car", "bench", "streetlight", "bus"]
ORNAMENT_EXAMPLES = ["lamp","vase","pillow","painting","plant","rug", "flower pots", "bird feeders", "trash bins", "traffic lights", "signs"]
PAIRING_RULES = ["chair near table","lamp on top of table/desk/counter","pillow on sofa/bed","vase centered on table", "bird feeders hanging on a tree", "trash bins beside bench"]
STEP_SIZE = 20

def extract_movement_from_note(env_note: str) -> Tuple[int,int]:
    note = (env_note or "").lower()
    dx, dy = 0, 0
    if "right" in note:
        dx, dy = 1, 0 
    elif "left" in note:
        dx, dy = -1, 0
    elif "backward" in note or "back" in note:
        dx, dy = 0, -1
    elif "forward" in note:
        dx, dy = 0, 1 
    return dx, dy


def _split_label(label):
    match = re.match(r"(.+?)_(\d+)$", label)
    if match:
        return match.group(1), int(match.group(2))
    else:
        return label, 0

def assign_progressive_ids(new_objects, scene_graph_dir):
    id_counter = {}

    for f in os.listdir(scene_graph_dir):
        if re.match(r"scene_pos.*\.json$", f):
            path = os.path.join(scene_graph_dir, f)
            with open(path, "r") as fp:
                try:
                    data = json.load(fp)
                    for obj in data.get("objects", []):
                        label = obj.get("label", "")
                        base_name, num = _split_label(label)
                        if base_name:
                            id_counter[base_name] = max(id_counter.get(base_name, 0), num)
                except json.JSONDecodeError:
                    print(f"⚠️ Skipping invalid JSON file: {f}")

    for obj in new_objects:
        base_name, _ = _split_label(obj.get("label", ""))
        next_id = id_counter.get(base_name, 0) + 1
        obj["label"] = f"{base_name}_{next_id}"
        id_counter[base_name] = next_id

    return new_objects

def normalize_obj(obj):
    if hasattr(obj, "dict"):
        return obj.dict()
    if isinstance(obj, str):
        return {"label": obj}
    return obj

def merge_delta(base_scene: dict, delta) -> dict:
    new_objs = [normalize_obj(obj) for obj in (delta.add or [])]
    remove_objs = [normalize_obj(obj) for obj in (delta.remove or [])]
    update_objs = [normalize_obj(obj) for obj in (delta.update or [])]

    scene_objects = [normalize_obj(o) for o in base_scene.get("objects", [])]

    if new_objs:
        scene_objects.extend(new_objs)

    if remove_objs:
        labels_to_remove = {obj.get("label") for obj in remove_objs if obj}
        scene_objects = [obj for obj in scene_objects if obj.get("label") not in labels_to_remove]

    if update_objs:
        update_map = {obj.get("label"): obj for obj in update_objs if obj}
        seen = set()
        for i, obj in enumerate(scene_objects):
            lbl = obj.get("label")
            if lbl in update_map:
                updated = {**obj, **update_map[lbl]}
                scene_objects[i] = updated
                seen.add(lbl)
        for lbl, upd in update_map.items():
            if lbl not in seen:
                scene_objects.append(upd)

    base_scene["objects"] = scene_objects
    return base_scene

def _parse_new_area_range(new_area_range: Optional[str]) -> Optional[Tuple[float, float, float, float]]:
    if not new_area_range:
        return None

    try:
        m_x = re.search(r"x\s*ranging\s*between\s*([-\d.]+)\s*\w*\s*and\s*([-\d.]+)", new_area_range, re.IGNORECASE)
        m_y = re.search(r"y\s*ranging\s*between\s*([-\d.]+)\s*\w*\s*and\s*([-\d.]+)", new_area_range, re.IGNORECASE)

        if not m_x:
            m_x = re.search(r"x\s*in\s*\(([-\d.]+),\s*([-\d.]+)\)", new_area_range, re.IGNORECASE)
        if not m_y:
            m_y = re.search(r"y\s*in\s*\(([-\d.]+),\s*([-\d.]+)\)", new_area_range, re.IGNORECASE)

        if m_x and m_y:
            return (
                float(m_x.group(1)),
                float(m_x.group(2)),
                float(m_y.group(1)),
                float(m_y.group(2)),
            )
    except Exception as e:
        print(f"Failed to parse range: {e}")

    return None

def _bbox_valid(bbox):
    if not isinstance(bbox, (list,tuple)) or len(bbox) != 4:
        return False
    x,y,w,h = bbox
    try:
        w = float(w); h = float(h)
    except Exception:
        return False
    return w > 0 and h > 0

def _dist(a, b):
    return ((a[0]-b[0])**2 + (a[1]-b[1])**2)**0.5

import re
from collections import Counter

def _parse_allowed_objects(instruction_text):
    allowed = Counter()

    match = re.search(r"Objects to be placed:\s*(.*?)(?:\.|\n|$)", instruction_text, re.IGNORECASE | re.DOTALL)
    if not match:
        print()
        return allowed

    object_section = match.group(1).strip()

    numbered = re.findall(r"(\d+)\s+([a-zA-Z_ ]+?)(?:,|$)", object_section)
    if numbered:
        for count, obj in numbered:
            obj_name = obj.strip().lower().rstrip("s")
            allowed[obj_name] += int(count)
        return allowed

    names = [o.strip().lower().rstrip("s") for o in re.split(r",| and ", object_section) if o.strip()]
    for obj_tmp in names:
        obj_name = "_".join(obj_tmp.strip().split())
        allowed[obj_name] += 1

    return allowed

def score_candidate_delta(delta, base_scene, allowed_objs, new_area_range_parsed, position, dx, dy):
    score = 0.0
    diagnostics = []

    print(allowed_objs)
    allowed_objs = {
    "_".join(k.strip().split()): v
    for k, v in allowed_objs.items()
}
    print(type(delta))
    print(delta)
    if delta and not isinstance(delta[0], dict):
        all_objs = [o.model_dump() if hasattr(o, "model_dump") else o.__dict__ for o in delta]
    else:
        all_objs = delta

    print(all_objs)

    print(new_area_range_parsed,"New area range parsed")
    if new_area_range_parsed:
        x0, x1, y0, y1 = new_area_range_parsed
        inside = 0
        outside_objects = []
        for o in all_objs:
            bbox = o.get("bbox")
            if not bbox or len(bbox) != 4:
                continue

            bx, by, bw, bh = bbox
            corners = [
                (bx, by),
                (bx + bw, by),
                (bx, by + bh),
                (bx + bw, by + bh)
            ]

            all_inside = all(x0 <= cx <= x1 and y0 <= cy <= y1 for cx, cy in corners)
            if all_inside:
                inside += 1
            else:
                outside_objects.append(o.get("label", "unknown_object"))

        if inside == len(all_objs):
            score += 1.0
        elif inside == 0:
            diagnostics.append("No added objects placed fully inside the new allowed area.")
            score -= 1.0

        if outside_objects:
            diagnostics.append(
                f"The following objects have bounding boxes extending outside the allowed area: "
                f"{', '.join(outside_objects)}."
            )
            score += 1- 1.0*len(outside_objects)/len(all_objs)
    
    print("Score after checking inside", score)

    c=0
    for o in all_objs:
        bx = o.get("bbox")
        if bx and isinstance(bx, (list, tuple)):
            if len(bx) >= 4:
                _, _, w, h = bx
                if w > 1000 or h > 1000:
                    diagnostics.append("unreasonably large bbox")
                    c -= 1.0/len(all_objs)
    score += 1 + c

    print("Bbox size score", score)

    predicted_labels = [
        re.sub(r"_\d+$", "", o.get("label", "").lower().replace(" ", "_").rstrip("s"))
        for o in all_objs
        if isinstance(o, dict) and "label" in o
    ]
    predicted_counts = Counter(predicted_labels)

    print(predicted_counts)
    print(allowed_objs)
    if allowed_objs:
        for obj_tmp, count in predicted_counts.items():
            obj = "_".join(obj_tmp.strip().split())
            if obj not in allowed_objs:
                diagnostics.append(f"Extra object '{obj}' found which is not part of the instructed list.")
                score -= 1.0/len(predicted_counts)
            elif count > allowed_objs[obj]:
                diagnostics.append(f"Too many '{obj}' objects (expected {allowed_objs[obj]}, found {count}).")
                score -= 1.0/len(predicted_counts)
        if predicted_counts == allowed_objs:
            diagnostics.append("All object counts match the instructed list perfectly.")
    score += 1.0

    print("After object missing", score)

    overlapping_pairs = []
    for i, o1 in enumerate(all_objs):
        b1 = o1.get("bbox")
        if not b1 or len(b1) < 4:
            continue

        x1, y1, w1, h1 = b1
        left1, right1 = x1, x1 + w1
        bottom1, top1 = y1, y1 + h1

        for j, o2 in enumerate(all_objs[i + 1:], start=i + 1):
            b2 = o2.get("bbox")
            if not b2 or len(b2) < 4:
                continue

            x2, y2, w2, h2 = b2
            left2, right2 = x2, x2 + w2
            bottom2, top2 = y2, y2 + h2

            overlap_x = not (right1 <= left2 or right2 <= left1)
            overlap_y = not (top1 <= bottom2 or top2 <= bottom1)

            if overlap_x and overlap_y:
                name1 = o1.get("name", o1.get("label", f"object_{i}"))
                name2 = o2.get("name", o2.get("label", f"object_{j}"))
                overlapping_pairs.append((name1, name2))

    n = len(all_objs)
    total_pairs = n * (n - 1) / 2 if n > 1 else 1
    overlap_count = len(overlapping_pairs)

    if overlap_count > 0:
        overlap_ratio = overlap_count / total_pairs
        pair_texts = [f"{a} ↔ {b}" for a, b in overlapping_pairs]
        diagnostics.append(
            f"Overlapping bounding boxes detected between: {', '.join(pair_texts)} "
            f"({overlap_count}/{int(total_pairs)} pairs overlap)."
        )
    else:
        overlap_ratio = 0.0
        diagnostics.append("No overlapping bounding boxes detected.")

    overlap_score = max(0.0, 1.0 - overlap_ratio)
    score += overlap_score

    print(f"Overlap ratio: {overlap_ratio:.2f}, overlap_score: {overlap_score:.2f}, total score: {score:.2f}")
    print("Overlapping Score", score)


    score/=4

    return score, diagnostics




def score_candidate_delta_one(delta, base_scene, allowed_objs, new_area_range_parsed, position, dx, dy):
    score = 0.0
    diagnostics = []

    print(allowed_objs)
    allowed_objs = {
    "_".join(k.strip().split()): v
    for k, v in allowed_objs.items()
}

    all_objs = []
    for o in delta:
        if hasattr(o, "model_dump"):
            all_objs.append(o.model_dump())
        elif isinstance(o, dict):
            all_objs.append(o)
        else:
            raise ValueError(f"Unexpected object type in delta: {type(o)}")

    if new_area_range_parsed:
        x0, x1, y0, y1 = new_area_range_parsed
        inside = 0
        outside_objects = []
        for o in all_objs:
            bbox = o.get("bbox")
            if not bbox or len(bbox) != 4:
                continue

            bx, by, bw, bh = bbox
            corners = [
                (bx, by),
                (bx + bw, by),
                (bx, by + bh),
                (bx + bw, by + bh)
            ]

            all_inside = all(x0 <= cx <= x1 and y0 <= cy <= y1 for cx, cy in corners)
            if all_inside:
                inside += 1
            else:
                outside_objects.append(o.get("label", "unknown_object"))

        if inside == len(all_objs):
            score += 1.0
        elif inside == 0:
            diagnostics.append("No added objects placed fully inside the new allowed area.")
            score -= 1.0

        if outside_objects:
            diagnostics.append(
                f"The following objects have bounding boxes extending outside the allowed area: "
                f"{', '.join(outside_objects)}."
            )
            score += 1- 1.0*len(outside_objects)/len(all_objs)
    
    print("Score after checking inside", score)

    c=0
    for o in all_objs:
        bx = o.get("bbox")
        if bx and isinstance(bx, (list, tuple)):
            if len(bx) >= 4:
                _, _, w, h = bx
                if w > 1000 or h > 1000:
                    diagnostics.append("unreasonably large bbox")
                    c -= 1.0/len(all_objs)
    score += 1 + c

    print("Bbox size score", score)

    predicted_labels = [
        re.sub(r"_\d+$", "", o.get("label", "").lower().replace(" ", "_").rstrip("s"))
        for o in all_objs
        if isinstance(o, dict) and "label" in o
    ]
    predicted_counts = Counter(predicted_labels)

    print(predicted_counts)
    print(allowed_objs)
    if allowed_objs:

        for obj_tmp, count in predicted_counts.items():
            obj = "_".join(obj_tmp.strip().split())
            if obj not in allowed_objs:
                diagnostics.append(f"Extra object '{obj}' found which is not part of the instructed list.")
                score -= 1.0/len(predicted_counts)
            elif count > allowed_objs[obj]:
                diagnostics.append(f"Too many '{obj}' objects (expected {allowed_objs[obj]}, found {count}).")
                score -= 1.0/len(predicted_counts)
        if predicted_counts == allowed_objs:
            diagnostics.append("All object counts match the instructed list perfectly.")
    score += 1.0

    overlapping_pairs = []
    for i, o1 in enumerate(all_objs):
        b1 = o1.get("bbox")
        if not b1 or len(b1) < 4:
            continue

        x1, y1, w1, h1 = b1
        left1, right1 = x1, x1 + w1
        bottom1, top1 = y1, y1 + h1

        for j, o2 in enumerate(all_objs[i + 1:], start=i + 1):
            b2 = o2.get("bbox")
            if not b2 or len(b2) < 4:
                continue

            x2, y2, w2, h2 = b2
            left2, right2 = x2, x2 + w2
            bottom2, top2 = y2, y2 + h2

            overlap_x = not (right1 <= left2 or right2 <= left1)
            overlap_y = not (top1 <= bottom2 or top2 <= bottom1)

            if overlap_x and overlap_y:
                name1 = o1.get("name", o1.get("label", f"object_{i}"))
                name2 = o2.get("name", o2.get("label", f"object_{j}"))
                overlapping_pairs.append((name1, name2))

    n = len(all_objs)
    total_pairs = n * (n - 1) / 2 if n > 1 else 1
    overlap_count = len(overlapping_pairs)

    if overlap_count > 0:
        overlap_ratio = overlap_count / total_pairs
        pair_texts = [f"{a} ↔ {b}" for a, b in overlapping_pairs]
        diagnostics.append(
            f"Overlapping bounding boxes detected between: {', '.join(pair_texts)} "
            f"({overlap_count}/{int(total_pairs)} pairs overlap)."
        )
    else:
        overlap_ratio = 0.0
        diagnostics.append("No overlapping bounding boxes detected.")

    overlap_score = max(0.0, 1.0 - overlap_ratio)
    score += overlap_score

    print(f"Overlap ratio: {overlap_ratio:.2f}, overlap_score: {overlap_score:.2f}, total score: {score:.2f}")
    print("Overlapping Score", score)


    score/=4

    return score, diagnostics

def extract_yes_no(verdict_text, question_number):
    pattern = rf"{question_number}\.?\s*[:\-]?\s*(yes|no|partially|maybe|not really|somewhat)"
    match = re.search(pattern, verdict_text.lower())
    if not match:
        return "unclear"
    answer = match.group(1)
    if "yes" in answer:
        return "yes"
    elif "no" in answer:
        return "no"
    else:
        return "unclear"


def load_existing_scenegraph(position, save_dir):
    x, y = position
    pattern = re.compile(rf"scene_pos_{x}_{y}\.json$")
    
    for fname in os.listdir(save_dir):
        if pattern.match(fname):
            fpath = os.path.join(save_dir, fname)
            with open(fpath, "r") as f:
                return json.load(f)
    return None

def visualize(scene, position, save_dir, save_name, sem_name):
    objects = []

    for obj in scene.get("objects", []):
        bbox = obj.get("bbox", [0, 0, 0, 0])
        tmp_object = {
            "text": obj.get("label", "unknown"),
            "x1": float(bbox[0]) - float(position[0]),
            "y1": float(bbox[1])- float(position[1]),
            "x2": float(bbox[0])+float(bbox[2])- float(position[0]),
            "y2": float(bbox[1])+float(bbox[3])- float(position[1])
        }
        objects.append(tmp_object)

    ego_pos = next((tuple(o["position"]) for o in scene.get("objects", []) if o.get("label") == "ego_person"), (0, 0))

    plot_utils.plot_bboxes(objects, position = position, env_name="scene", save_dir=save_dir, save_name= save_name)
    plot_utils.plot_semplot(objects, position = position, env_name="scene", save_dir=save_dir, save_name= sem_name)


def plot_scenegraph(scenegraph, position, save_dir):
    G = nx.DiGraph()
    objects = scenegraph.get("objects", [])

    label_colors = {
        l: (random.random(), random.random(), random.random())
        for l in set(o["label"] for o in objects if o["label"] != "ego_person")
    }

    ego_obj = next((o for o in objects if o.get("label") == "ego_person"), None)
    if ego_obj:
        ego_pos = ego_obj.get("position", [0, 0])
    else:
        ego_pos = [0, 0]
    G.add_node("ego_person", pos=tuple(ego_pos), color=(0, 1, 0))

    for obj in objects:
        label = obj["label"]
        dx, dy = obj.get("position", [0, 0])
        color = (0, 1, 0) if label == "ego_person" else label_colors[label]

        G.add_node(label, pos=(dx, dy), color=color)

        if label != "ego_person":
            G.add_edge("ego_person", label)

    pos_dict = nx.get_node_attributes(G, "pos")
    colors = [G.nodes[n]["color"] for n in G.nodes()]

    plt.figure(figsize=(8, 6))
    nx.draw(G, pos_dict, with_labels=True, node_color=colors, node_size=500)
    plt.savefig(os.path.join(save_dir, f"pos_{position[0]}_{position[1]}.png"))
    plt.clf()
    plt.close('all')