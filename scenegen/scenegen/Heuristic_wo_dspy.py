# # """
# # SceneCritic testbed with heuristic as a Critic for refinement  (NON-DSPy version)

# # DSPy -> plain-Python port
# # -------------------------
# # * dspy.LM / dspy.JSONAdapter     -> litellm.completion + a JSON parser.
# # * dspy.Signature                 -> a prompt template ("build_io_prompt") that
# #                                     lists inputs and demands a JSON object keyed
# #                                     by the old OutputField names.
# # * dspy.ChainOfThought(Signature) -> a normal class whose forward() builds the
# #                                     prompt, calls the LM, and parses the JSON.
# # * dspy.Refine(...)               -> an explicit N-attempt loop in
# #                                     ObjectPlacementManager that re-uses
# #                                     placement_reward() to score + build feedback.
# # * Pydantic SceneObject instances are still returned by the placement stage so
# #   downstream `.model_dump()` / `utils.normalize_obj(...)` calls behave as before.

# # The OrientationRefinerModule / ScaleRefinerModule remain unimplemented stubs,
# # exactly as in the original testbed file.

# # If you serve the model with a local OpenAI-compatible endpoint (vLLM/SGLang),
# # edit the LLM CONFIG block below (e.g. LLM_MODEL="openai/TIGER-Lab/VL-Reasoner-72B",
# # LLM_API_BASE="http://127.0.0.1:7501/v1").
# # """

# # import os
# # import sys
# # import re
# # import io
# # import copy
# # import json
# # import base64
# # from typing import Any, Dict, List, Optional

# # import matplotlib.pyplot as plt
# # import networkx as nx
# # import numpy as np
# # from matplotlib.offsetbox import AnnotationBbox, OffsetImage
# # from matplotlib.patches import Rectangle
# # from PIL import Image
# # from pydantic import BaseModel, Field

# # import litellm  # <-- replaces dspy/litellm-via-dspy

# # sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# # import testbed.plot_utilities as plot_utilities
# # import testbed.utils as utils


# # # =============================================================================
# # # LLM CONFIG
# # # =============================================================================
# # # LLM_MODEL = "openai/Qwen/Qwen3.5-27B"
# # # LLM_API_KEY = "local"          # mirrors dspy.LM(api_key='')
# # # LLM_API_BASE = "http://127.0.0.1:8091/v1"       # set e.g. "http://127.0.0.1:7501/v1" for a local vLLM/SGLang server
# # # LLM_TEMPERATURE = 0.5

# # LLM_MODEL    = "openai/Qwen/Qwen3.5-27B"
# # LLM_API_KEY  = "local"                       # non-empty (empty gets dropped → OpenAI client rejects)
# # LLM_API_BASE = "http://127.0.0.1:7501/v1"
# # LLM_TEMPERATURE = 0.5

# # JSON_SYSTEM_PROMPT = (
# #     "You are a precise assistant that outputs ONLY a single valid JSON object. "
# #     "Do not include markdown code fences, comments, reasoning, or any prose "
# #     "outside the JSON object."
# # )


# # def _build_messages(user_prompt: str, images=None, system: str = JSON_SYSTEM_PROMPT):
# #     """Build an OpenAI/litellm-style messages list, optionally multimodal."""
# #     msgs = []
# #     if system:
# #         msgs.append({"role": "system", "content": system})
# #     if images:
# #         content = [{"type": "text", "text": user_prompt}]
# #         for b64 in images:
# #             content.append({
# #                 "type": "image_url",
# #                 "image_url": {"url": f"data:image/png;base64,{b64}"},
# #             })
# #         msgs.append({"role": "user", "content": content})
# #     else:
# #         msgs.append({"role": "user", "content": user_prompt})
# #     return msgs


# # def call_llm_raw(user_prompt: str, images=None, system: str = JSON_SYSTEM_PROMPT,
# #                  temperature: float = LLM_TEMPERATURE) -> str:
# #     """Single raw completion call through litellm."""
# #     kwargs = dict(
# #         model=LLM_MODEL,
# #         messages=_build_messages(user_prompt, images=images, system=system),
# #         temperature=temperature,
# #     )
# #     if LLM_API_KEY:
# #         kwargs["api_key"] = LLM_API_KEY
# #     if LLM_API_BASE:
# #         kwargs["api_base"] = LLM_API_BASE
# #     resp = litellm.completion(**kwargs)
# #     return resp.choices[0].message.content


# # def call_llm(user_prompt: str, images=None, system: str = JSON_SYSTEM_PROMPT,
# #              temperature: float = LLM_TEMPERATURE, retries: int = 2):
# #     """
# #     Call the LM and parse a JSON object/array out of its reply.
# #     Returns (parsed_or_None, raw_text_of_last_attempt).
# #     """
# #     last_raw = None
# #     for _ in range(retries + 1):
# #         last_raw = call_llm_raw(user_prompt, images=images, system=system, temperature=temperature)
# #         parsed = parse_json_response(last_raw)
# #         if parsed is not None:
# #             return parsed, last_raw
# #     return None, last_raw


# # def pil_to_b64(img: Image.Image) -> str:
# #     buf = io.BytesIO()
# #     img.convert("RGB").save(buf, format="PNG")
# #     return base64.b64encode(buf.getvalue()).decode("utf-8")


# # # =============================================================================
# # # JSON cleaning / extraction helpers
# # # =============================================================================
# # def clean_model_output(raw_text: str) -> str:
# #     text = re.sub(r"```(?:json)?", "", raw_text, flags=re.I)
# #     text = re.sub(r"^(system|user|assistant|You are a helpful assistant).*",
# #                   "", text, flags=re.I | re.MULTILINE)

# #     lines = []
# #     for line in text.splitlines():
# #         line = line.strip()
# #         if not line:
# #             continue
# #         if line.startswith(("⚠️", "Do not", "Perform")):
# #             continue
# #         lines.append(line)
# #     return "\n".join(lines)


# # def extract_json(raw_text: str) -> Dict[str, Any]:
# #     cleaned = clean_model_output(raw_text)
# #     try:
# #         data = json.loads(cleaned)
# #         if isinstance(data, dict) and any(k in data for k in ("atomic_tasks", "objects", "scene")):
# #             return data
# #     except json.JSONDecodeError:
# #         pass

# #     stack = []
# #     start_idx = None
# #     for i, c in enumerate(cleaned):
# #         if c == "{":
# #             if not stack:
# #                 start_idx = i
# #             stack.append("{")
# #         elif c == "}":
# #             if stack:
# #                 stack.pop()
# #                 if not stack and start_idx is not None:
# #                     candidate = cleaned[start_idx:i + 1]
# #                     try:
# #                         data = json.loads(candidate)
# #                         if any(k in data for k in ("atomic_tasks", "objects", "scene")):
# #                             return data
# #                     except json.JSONDecodeError:
# #                         continue
# #     return {}


# # def parse_json_response(raw_text: str):
# #     """Return the first balanced JSON object or array found in the LM output, or None."""
# #     if raw_text is None:
# #         return None
# #     cleaned = clean_model_output(raw_text)

# #     try:
# #         return json.loads(cleaned)
# #     except json.JSONDecodeError:
# #         pass

# #     for open_c, close_c in (("{", "}"), ("[", "]")):
# #         depth = 0
# #         start = None
# #         for i, c in enumerate(cleaned):
# #             if c == open_c:
# #                 if depth == 0:
# #                     start = i
# #                 depth += 1
# #             elif c == close_c and depth > 0:
# #                 depth -= 1
# #                 if depth == 0 and start is not None:
# #                     candidate = cleaned[start:i + 1]
# #                     try:
# #                         return json.loads(candidate)
# #                     except json.JSONDecodeError:
# #                         break
# #     return None


# # def get_list(resp, keys):
# #     """Pull a list out of an LM response under one of `keys` (or the root)."""
# #     if isinstance(resp, list):
# #         return resp
# #     if isinstance(resp, dict):
# #         for k in keys:
# #             v = resp.get(k)
# #             if isinstance(v, list):
# #                 return v
# #         scene = resp.get("scene")
# #         if isinstance(scene, dict):
# #             for k in keys:
# #                 v = scene.get(k)
# #                 if isinstance(v, list):
# #                     return v
# #     return []


# # # =============================================================================
# # # Object Classes  (unchanged)
# # # =============================================================================
# # class SceneObject(BaseModel):
# #     label: str = Field(..., description="Name of the object (e.g. 'chair').")
# #     object_type: str = Field(..., description="Choose 'bulk' or 'ornaments'.")
# #     anchor: Optional[str] = Field(None, description="Reference object for spatial relation, if any.")
# #     bbox: List[float] = Field(..., min_length=4, max_length=4,
# #                               description="Bounding box [x,y,w,h] in scene coordinates in meters.")
# #     position: List[float] = Field(..., min_length=2, max_length=2,
# #                                   description="Object center [x+w/2,y+h/2] in scene coordinates in meters.")
# #     rotation_angle: Optional[float] = Field(0.0, description="Object rotation in degrees.")
# #     description: str = Field(..., description="Natural language description of the object.")
# #     concise_description: str = Field(..., description="Natural language description of only the object without mentioning its relation with other objects.")


# # class OutputSchema(BaseModel):
# #     add: Optional[List[SceneObject]] = Field(default_factory=list,
# #                                              description="Objects to add to the scene.")
# #     remove: Optional[List[str]] = Field(default_factory=list,
# #                                         description="Labels of objects to remove.")
# #     update: Optional[List[SceneObject]] = Field(default_factory=list,
# #                                                 description="Objects to update (position, bbox, etc.).")


# # class AtomicTask(BaseModel):
# #     id: str = Field(..., description="Unique identifier for the task.")
# #     object_type: str = Field(..., description="Choose 'bulk' or 'ornaments'.")
# #     type: str = Field(..., description="Task type: 'Add', 'Remove', or 'Update'.")
# #     label: str = Field(..., description="Label of the object this task refers to.")
# #     anchor: Optional[str] = Field(None, description="Reference object for spatial relation, if any.")
# #     spatial_hint: Optional[str] = Field(None, description="Human-readable spatial instruction.")
# #     bbox_hint: Optional[List[int]] = Field(None, description="Optional bounding box hint [x,y,w,h].")
# #     output_schema: OutputSchema = Field(..., description="Structured delta to apply to the scene graph.")
# #     execution_prompt: str = Field(..., description="Concise LLM instruction to run this task.")


# # def coerce_scene_object(d) -> Optional[SceneObject]:
# #     """Tolerantly build a SceneObject from an LM dict, filling sane defaults."""
# #     if not isinstance(d, dict):
# #         return None
# #     label = d.get("label")
# #     if not label:
# #         return None

# #     bbox = d.get("bbox")
# #     if not (isinstance(bbox, (list, tuple)) and len(bbox) == 4):
# #         pos = d.get("position")
# #         if isinstance(pos, (list, tuple)) and len(pos) == 2:
# #             bbox = [pos[0] - 0.5, pos[1] - 0.5, 1.0, 1.0]
# #         else:
# #             bbox = [0.0, 0.0, 1.0, 1.0]
# #     try:
# #         bbox = [float(v) for v in bbox]
# #     except (TypeError, ValueError):
# #         bbox = [0.0, 0.0, 1.0, 1.0]

# #     pos = d.get("position")
# #     if not (isinstance(pos, (list, tuple)) and len(pos) == 2):
# #         pos = [bbox[0] + bbox[2] / 2.0, bbox[1] + bbox[3] / 2.0]
# #     try:
# #         pos = [float(pos[0]), float(pos[1])]
# #     except (TypeError, ValueError):
# #         pos = [bbox[0] + bbox[2] / 2.0, bbox[1] + bbox[3] / 2.0]

# #     desc = d.get("description") or str(label)
# #     concise = d.get("concise_description") or desc

# #     rot = d.get("rotation_angle", 0.0)
# #     try:
# #         rot = float(rot) if rot is not None else 0.0
# #     except (TypeError, ValueError):
# #         rot = 0.0

# #     try:
# #         return SceneObject(
# #             label=str(label),
# #             object_type=str(d.get("object_type", "bulk")),
# #             anchor=d.get("anchor"),
# #             bbox=bbox,
# #             position=pos,
# #             rotation_angle=rot,
# #             description=str(desc),
# #             concise_description=str(concise),
# #         )
# #     except Exception as e:  # noqa: BLE001
# #         print(f"[coerce_scene_object] skipping invalid object {label!r}: {e}")
# #         return None


# # def to_scene_objects(items) -> List[SceneObject]:
# #     out = []
# #     for it in items or []:
# #         obj = coerce_scene_object(it)
# #         if obj is not None:
# #             out.append(obj)
# #     return out


# # # =============================================================================
# # # Generic prompt builder for "signature-like" calls
# # # =============================================================================
# # def build_io_prompt(instructions: str, inputs: Dict[str, Any], outputs: Dict[str, str]) -> str:
# #     parts = []
# #     if instructions:
# #         parts.append(instructions.strip())
# #     parts.append("\n--- INPUTS ---")
# #     for name, val in inputs.items():
# #         parts.append(f"\n[{name}]:\n{val}")
# #     parts.append("\n--- OUTPUT FORMAT ---")
# #     parts.append("Return ONLY one JSON object with exactly these top-level keys (no extra text, no markdown):")
# #     for name, desc in outputs.items():
# #         parts.append(f'  "{name}": {desc}')
# #     return "\n".join(parts)


# # # =============================================================================
# # # Prompt Building  (unchanged from the original)
# # # =============================================================================
# # def build_planning_prompt(position, env_note, prev_scenegraphs, save_dir, dim_x, dim_y, dx, dy, object_store, scene_desc=None, ornaments_included=False):
# #     x, y = position
# #     role_block = f"Role: You are an expert layout planner.\n"

# #     env_block = f"""
# #     Environment note: {env_note}
# #     """

# #     placement_order = f"""
# #     - Objects to be placed: {object_store}. Do not place any other objects than this list.
# #     - DO NOT PLACE ANY ORNAMENTS.
# #     - You have to place all the objects mentioned.
# #     """

# #     object_check = f"""
# #     Strictly follow these rules:
# #     - Objects not to include: 
# #         - windows, doors, walls, partitions or anything attached to it.
# #         - floors, roofs.
# #         - roads, street, pavement, footpath, sidewalk, pathway, lawn and any other part of ground.
# #         - objects like counter that has multiple english meanings.
# #     - Include only specific objects, not generalized names like new_object, appliances etc.
# #     - Do not include any objects in plural.
# #     """

# #     prev_objects = []
# #     for sg in prev_scenegraphs:
# #         if isinstance(sg, dict):
# #             if "scene" in sg and isinstance(sg["scene"], dict):
# #                 prev_objects.extend(sg["scene"].get("objects", []))
# #             else:
# #                 prev_objects.extend(sg.get("objects", []))

# #     prev_objects_summary = ", ".join([
# #         f"{obj.get('label','unknown')} at {obj.get('position',[0,0])}"
# #         for obj in prev_objects if obj.get("label") != "ego_person"
# #     ])

# #     if prev_objects:
# #         if dy != 0:
# #             dire = f"move along Y-axis by {dim_y} meters"
# #             y = [position[1], position[1] + dim_y]
# #             x = [position[0] - dim_x // 2, position[0] + dim_x // 2]
# #             new_area_range = f"x ranging between {x[0]} meters and {x[1]} meters, y ranging between {y[0]} meters and {y[1]} meters"
# #         elif dx != 0:
# #             dire = f"move along X-axis by {dim_x} meters"
# #             y = [position[1], position[1] + dim_y]
# #             x = [position[0] - dim_x // 2, position[0] + dim_x // 2]
# #             new_area_range = f"x ranging between {x[0]} meters and {x[1]} meters, y ranging between {y[0]} meters and {y[1]} meters"
# #         else:
# #             dire = f"stay at same"
# #             new_area_range = None

# #         movement_block = f"""
# #     Previous scene objects: {prev_objects_summary}
# #     Movement Note: You {dire} to absolute ({position}).
# #     """
# #         consistency_block = f"""
# #     Consistency Note:
# #     - Objects mentioned in instruction should appear.
# #     """
# #     else:
# #         x = [position[0] - dim_x // 2, position[0] + dim_x // 2]
# #         y = [position[1], position[1] + dim_y]
# #         new_area_range = f"x ranging between {x[0]} meters and {x[1]} meters, y ranging between {y[0]} meters and {y[1]} meters"
# #         movement_block = f"Movement Note: You are at {position}."
# #         consistency_block = f"""
# #     Consistency Note:
# #     - Initialize scene with new objects ONLY in the visible area: {new_area_range}.
# #     - Generate a scene description.
# #     """

# #     planning_prompt = role_block + env_block + placement_order + object_check

# #     if scene_desc:
# #         scene_desc_block = f"""
# #         New Scene Description:
# #         {scene_desc}
# #         """
# #         planning_prompt = planning_prompt + scene_desc_block

# #     semantic_block = f"""
# #     Object Arrangement and Spatial Relationships:
# #     - Objects should be arranged naturally as seen in real world environment (e.g., chairs around table, cars parked along street, trees spaced in a park, lamp on top of table/desk, pillow on sofa/bed , television facing sofa , vase centered on table ,  bird feeders hanging on a tree, trash bins beside bench ).
# #     - Define clear spatial relationships between objects (e.g., next to, under, near, on top of, centered on, beside).
# #     - Align most larger objects along area boundaries(e.g. walls, fences or paths) and corners to reflect natural real-world arrangements and maximise navigable spaces.
# #     - Ensure bounding boxes and positions are consistent with the defined spatial relationships.
    
# #     Follow semantic asset grouping:
# #     - A semantic asset group is a collection of assets that are logically, functionally, or spatially related within a scene. Assets in a group support a common purpose or activity and together form a meaningful subscene. 
# #     - Group assets based on function (used together), semantics (conceptually related), or geometry (close in space or aligned).
# #     - Examples:
# #         - Sleeping area: Bed + Nightstand + Lamp (used together for resting and reading)
# #         - Entertainment area: Sofa + TV Console + TV  (functionally related for watching)
# #         - Work area: Desk + Chair + Laptop (supports the working activity)
# #         - Dining area: Dining Table + Chairs + Chandelier (forms a cohesive eating zone)
# #     """

# #     task_block = """
# # Task:
# # Decompose this instruction into ATOMIC TASKS for the design process. The output should ONLY be a valid json."""
# #     planning_prompt = planning_prompt + semantic_block + task_block

# #     file_path = os.path.join(save_dir, f"planning_prompt_{position[0]}_{position[1]}.txt")
# #     with open(file_path, "w", encoding="utf-8") as f:
# #         f.write(planning_prompt)

# #     return planning_prompt, new_area_range, consistency_block, movement_block


# # def build_object_placement_prompt(plan_text, position, dim_x, dim_y, consistency_block, movement_block, env_note, new_area_range):
# #     role_block = f"Role: You are an expert real world planner.\n"
# #     env_block = f"Environment note: {env_note}.\n"
# #     return role_block + env_block + f"""

# # Plan summary for the room:
# # {plan_text}

# # Placement Rules:
# # - You are populating a {dim_x} meters * {dim_y} meters room.
# # - The room dimensions are x ranging between -{dim_x/2} to {dim_x/2} and y ranging between 0 to {dim_y}. Bounding boxes SHOULD NOT go out of the room area.
# # - Design such that two BULK objects bounding boxes SHOULD NOT INTERSECT each other unless the relationship is under or on top of.
# # - While placing, verify that the front of the object being placed and already placed objects remains clearly visible and is not obstructed by any larger object positioned in front of it.
# # - Maintain realistic distances between the objects you are placing to make the room visually appealing. Do not clutter them together.
# # - Align most larger objects along area boundaries(e.g. walls, fences or paths) to reflect natural real-world arrangements and maximise navigable spaces.
# # - Each object must include: label(same as the name mentioned in plan, do not add any adjective), object_type, anchor, description, bbox [x,y,w,h] where (x+w/2,y+h/2) is the position/center, (x,y) is the bottom left corner and (w,h) is the width and height, description, position [x+w/2,y+h/2], rotation angle.
# # - Object sizes and bounding boxes must be semantically consistent with other objects in the scene and the room dimension.
# # - Positions and bounding boxes must accurately reflect spatial relationships (e.g., next to, under, on top of, centered on, near).
# # - Anchored objects must be placed and oriented relative to their reference object according to the defined spatial relationship.

# # Instructions:
# # - Output ONLY valid JSON with fields: objects.


# # """


# # def build_orientation_prompt(scene: Dict[str, Any]) -> str:
# #     objs = scene.get("objects", [])
# #     obj_lines = "\n".join([
# #         f"- {o['label']}: {o.get('description', '')} (anchor: {o.get('anchor', 'none')})"
# #         for o in objs
# #     ])

# #     return f"""
# # You are refining object orientations in a 3D indoor or outdoor scene.
# # Assign realistic rotation angles (in degrees, 0–360) to each object.

# # Interpretation:
# # - 0°: facing forward (toward the ego or camera)
# # - 90°: facing right
# # - 180°: facing backward
# # - 270°: facing left

# # Guidelines:
# # - Objects attached to walls should face away from the wall.
# # - Sofas and chairs should face TVs or tables if present.
# # - Tables should align with nearby seating.
# # - TVs and monitors should face the main sitting area.
# # - Beds should face away from their headboard wall.
# # - Cabinets, wardrobes, and shelves should be against walls and face outward.
# # - Appliances (stoves, sinks, fridges) should face the open workspace or usable area.
# # - Floor lamps, indoor plants, and static decor items usually have 0° rotation.
# # - Benches should face open spaces, walkways, or scenic views.
# # - Vehicles should face their direction of travel or toward open space.
# # - Trees, rocks, and natural outdoor elements should maintain 0° rotation.
# # - Freestanding objects in open areas should face the scene center or main viewpoint.
# # - Groups of similar objects (e.g., multiple chairs) should be aligned symmetrically.
# # - Desks should face open space or toward the user's seating direction.
# # - Bathroom fixtures (sinks, toilets) should face the center or entry of the room, not walls.

# # Scene objects:
# # {scene}

# # - Output ONLY valid JSON with fields: objects exactly similar to input. Do not output reasoning.
# # """.strip()


# # def build_scale_prompt(scene: Dict[str, Any]) -> str:
# #     print(scene, "Scene build")
# #     objs = scene.get("objects", [])
# #     obj_labels = ", ".join([o["label"] for o in objs])

# #     return f"""
# # You are refining object scales in a realistic 3D scene.

# # Adjust the size of each object's bounding box based on its natural proportion relative to surrounding objects in the scene. Maintain realistic scale relationships as seen in real environments.
# # Ensure relative size consistency — smaller items (e.g., lamps, books, vases) should not exceed larger reference objects (e.g., tables, beds, sofas).
# # Examples:
# # - A chair should be smaller than a desk/table.
# # - A desk should be smaller than a bed.
# # - A nightstand should be lower than a bed.
# # - A sofa should be larger than a side table.
# # - Trees should be taller than nearby benches or cars.
# # - Buildings should dwarf trees and vehicles.
# # Keep proportions coherent within the environment (indoor or outdoor), adjusting bounding box dimensions for necessary objects so all objects appear physically plausible when viewed together.

# # Scene objects: 
# # {scene}

# # - Output ONLY valid JSON with fields: objects exactly similar to input.
# # """.strip()


# # # =============================================================================
# # # Modules  (plain Python, no DSPy)
# # # =============================================================================
# # class SceneDescription:
# #     """Replaces dspy.ChainOfThought(SceneDescriptionGenerator)."""

# #     def __call__(self, **kwargs):
# #         return self.forward(**kwargs)

# #     def forward(self, env_note: str, scene_desc: str):
# #         print(f"\n Generating new scene description for position ")
# #         instructions = (
# #             "Generate a concise updated description of the current scene using "
# #             "prior context, environment notes, and predicted area type."
# #         )
# #         prompt = build_io_prompt(
# #             instructions,
# #             inputs={
# #                 "random_seed": utils.generate_random_seed(),
# #                 "previous_scene_description": scene_desc,
# #                 "env_note": env_note,
# #             },
# #             outputs={
# #                 "new_scene_description": "string — a short, coherent textual description of the new scene.",
# #                 "concise_scene_name": "string — one-word name of the area you are generating.",
# #                 "scene_area": "string — either 'indoor' or 'outdoor'.",
# #             },
# #         )
# #         parsed, _ = call_llm(prompt)
# #         parsed = parsed or {}
# #         new_scene_desc = parsed.get("new_scene_description", "No scene generated.")
# #         new_scene_name = parsed.get("concise_scene_name", "No scene generated.")
# #         new_scene_area = parsed.get("scene_area", "No scene generated.")
# #         return new_scene_desc, new_scene_name, new_scene_area


# # class ObjectChoice:
# #     """Replaces dspy.ChainOfThought(ObjectChoicePlanner)."""

# #     def suggest_objects(self, env_note: str, scene_description: str) -> Dict[str, List[str]]:
# #         choice_restriction_prompt = """
# #             Strictly follow these rules:
# # - Objects not to include: 
# #     - windows, doors, walls, partitions or anything attached to it.
# #     - floors, roofs.
# #     - roads, street, pavement, footpath, sidewalk, pathway,lawn and any other part of ground.
# #     - objects like counter that has multiple english meanings.
# # - Include only specific objects, not generalized names like new_object, appliances etc.
# # - Do not include any objects in plural.
# # - Do not repeat any object more than twice. Try to avoid repeating objects.
# #             """
# #         instructions = "Decide potential bulk and ornament objects for the described scene."
# #         prompt = build_io_prompt(
# #             instructions,
# #             inputs={
# #                 "random_seed": utils.generate_random_seed(),
# #                 "env_note": env_note,
# #                 "scene_description": scene_description,
# #                 "choice_restriction_prompt": choice_restriction_prompt,
# #             },
# #             outputs={
# #                 "bulk_objects": "JSON list of strings — large/structural/fixed objects (e.g. building, tree, table, sofa, road).",
# #                 "ornament_objects": "JSON list of strings — smaller/decorative/movable objects (e.g. lamp, plant, trash bin, signboard).",
# #             },
# #         )
# #         parsed, _ = call_llm(prompt)
# #         parsed = parsed or {}
# #         bulk = parsed.get("bulk_objects") or []
# #         ornaments = parsed.get("ornament_objects") or []
# #         if not isinstance(bulk, list):
# #             bulk = []
# #         if not isinstance(ornaments, list):
# #             ornaments = []
# #         return {"bulk_objects": bulk, "ornament_objects": ornaments}


# # class PlanAtomicTasks:
# #     """Replaces dspy.ChainOfThought(AtomicTaskPlanning)."""

# #     def __init__(self, save_dir):
# #         self.save_dir = save_dir

# #     def __call__(self, **kwargs):
# #         return self.forward(**kwargs)

# #     def forward(self, position, env_note, scene_desc, object_store, prev_scenegraphs, dim_x, dim_y, dx, dy):
# #         planning_prompt, new_area_range, consistency_note, movement_note = build_planning_prompt(
# #             position=position, env_note=env_note, prev_scenegraphs=prev_scenegraphs,
# #             save_dir=self.save_dir, dim_x=dim_x, dim_y=dim_y, dx=dx, dy=dy,
# #             object_store=object_store, scene_desc=scene_desc, ornaments_included=False,
# #         )

# #         allowed_objs = utils._parse_allowed_objects(planning_prompt)
# #         print("Inside planner")

# #         atomic_task_spec = (
# #             'JSON list of atomic-task objects. Each element MUST contain keys: '
# #             '"id" (string), "object_type" ("bulk" or "ornaments"), '
# #             '"type" ("Add"/"Remove"/"Update"), "label" (string), '
# #             '"anchor" (string or null), "spatial_hint" (string or null), '
# #             '"bbox_hint" ([x,y,w,h] ints or null), '
# #             '"output_schema" (object with "add"/"remove"/"update" lists), '
# #             '"execution_prompt" (string — concise instruction to run this task).'
# #         )
# #         prompt = (
# #             planning_prompt
# #             + "\n\n--- OUTPUT FORMAT ---\n"
# #             + f"random_seed: {utils.generate_random_seed()}\n"
# #             + 'Return ONLY one JSON object: {"atomic_tasks": [ ... ]} where each element is described as:\n'
# #             + atomic_task_spec
# #         )

# #         parsed, _ = call_llm(prompt)
# #         print("After planning computation")

# #         raw_tasks = get_list(parsed, ["atomic_tasks", "atomic_tasks_json"])

# #         normalized_tasks = []
# #         for t in raw_tasks:
# #             if not isinstance(t, dict):
# #                 continue
# #             try:
# #                 normalized_tasks.append(AtomicTask(**t).model_dump())
# #             except Exception:
# #                 normalized_tasks.append(t)

# #         parsed_pred = {"atomic_tasks": normalized_tasks}

# #         with open(os.path.join(self.save_dir, f"planning_{position[0]}_{position[1]}.json"), "w") as f:
# #             json.dump(parsed_pred, f, indent=2)
# #         return parsed_pred, new_area_range, consistency_note, movement_note, allowed_objs


# # class ObjectPlacementManager:
# #     """
# #     Replaces dspy.ChainOfThought(ObjectPlacementGenerator) wrapped in dspy.Refine.
# #     Runs up to N placement attempts, scoring each with placement_reward() and
# #     feeding the diagnostics back as feedback, keeping the best-scoring attempt.
# #     """

# #     def __init__(self, save_dir):
# #         self.save_dir = save_dir
# #         self.N = 5
# #         self.threshold = 1.0
# #         self._refine_logs = []

# #     def __call__(self, *args, **kwargs):
# #         return self.forward(*args, **kwargs)

# #     def _generate_placement(self, instruction_prompt: str, feedback: str) -> List[SceneObject]:
# #         object_spec = (
# #             'Each element MUST contain keys: '
# #             '"label" (string, identical to the plan name, no adjectives), '
# #             '"object_type" ("bulk" or "ornaments"), '
# #             '"anchor" (string or null), '
# #             '"bbox" ([x,y,w,h] floats; (x,y) bottom-left corner, (w,h) width/height), '
# #             '"position" ([x+w/2, y+h/2] floats), '
# #             '"rotation_angle" (float degrees), '
# #             '"description" (string), '
# #             '"concise_description" (string describing only the object, no relations).'
# #         )
# #         user = instruction_prompt
# #         user += f"\n\nrandom_seed: {utils.generate_random_seed()}\n"
# #         if feedback:
# #             user += "\nFeedback based on previous placements (follow it):\n" + feedback + "\n"
# #         user += (
# #             '\n--- OUTPUT FORMAT ---\n'
# #             'Return ONLY one JSON object: {"objects": [ ... ]}.\n' + object_spec
# #         )

# #         parsed, _ = call_llm(user)
# #         items = get_list(parsed, ["objects", "placement_json"])
# #         return to_scene_objects(items)

# #     def forward(self, position, plan, new_area_range, allowed_objs, dim_x, dim_y,
# #                 consistency_block, movement_block, base_scene, env_note="No note"):
# #         plan_text = "\n".join(
# #             task.get('execution_prompt', '')
# #             for task in plan.get("atomic_tasks", [])
# #             if task.get('execution_prompt')
# #         )
# #         prompt = build_object_placement_prompt(
# #             plan_text, position, dim_x, dim_y, consistency_block, movement_block, env_note, new_area_range
# #         )

# #         # dx / dy come from the module-global scope (set in __main__), matching the
# #         # original code which referenced them as globals inside executor_args.
# #         args = {
# #             "instruction_prompt": prompt,
# #             "base_scene": base_scene,
# #             "new_area_range": new_area_range,
# #             "position": position,
# #             "dx": dx,
# #             "dy": dy,
# #             "allowed_objs": allowed_objs,
# #             "feedback": "",
# #         }

# #         # ----- manual refinement loop (replaces dspy.Refine) -----
# #         best_delta: List[SceneObject] = []
# #         best_score = float("-inf")

# #         for _ in range(self.N):
# #             delta = self._generate_placement(args["instruction_prompt"], args["feedback"])
# #             score = self.placement_reward(args, delta)  # also mutates args["feedback"]

# #             if score > best_score:
# #                 best_score = score
# #                 best_delta = delta

# #             if score >= self.threshold:
# #                 best_delta = delta
# #                 break
# #         # ---------------------------------------------------------

# #         delta = best_delta

# #         delta_objects = [utils.normalize_obj(o) for o in delta]

# #         if delta_objects:
# #             delta_objects = utils.assign_progressive_ids(
# #                 delta_objects,
# #                 scene_graph_dir=self.save_dir
# #             )

# #         base_scene["objects"].extend(delta_objects)
# #         return base_scene

# #     def placement_reward(self, args, pred):
# #         tmp_scene = copy.deepcopy(args.get("base_scene"))
# #         delta = getattr(pred, "placement_json", pred)
# #         delta_objects = [utils.normalize_obj(o) for o in delta]
# #         tmp_scene["objects"].extend(delta_objects)
# #         utils.visualize(tmp_scene, position=pos, save_dir=save_dir,
# #                         save_name=f"check_{len(self._refine_logs) + 1}",
# #                         sem_name=f"sem_plot_{len(self._refine_logs) + 1}")

# #         base_scene = args.get("base_scene")
# #         new_area_range = args.get("new_area_range")
# #         new_area_range_parsed = utils._parse_new_area_range(new_area_range)
# #         position = args.get("position", (0, 0))
# #         dx_ = args.get("dx", 0)
# #         dy_ = args.get("dy", 0)
# #         allowed_objs = args.get("allowed_objs")

# #         score, diagnostics = utils.score_candidate_delta(
# #             delta, base_scene, allowed_objs, new_area_range_parsed, position, dx_, dy_
# #         )

# #         feedback_text = (
# #             "Placement Diagnostics / Issues:\n"
# #             + "\n".join(f"- {d}" for d in diagnostics)
# #             + "\n\nAction Required:\n"
# #             "Update the positions or bounding box dimensions of the objects having placement problems to address the above issues. "
# #             "Ensure that any changes do not cause bounding box overlaps with other objects in the scene."
# #         )

# #         args["feedback"] = feedback_text
# #         log_path = os.path.join(self.save_dir, f"refine_logs__{position[0]}_{position[1]}.json")
# #         self._refine_logs.append({
# #             "iteration": len(self._refine_logs) + 1,
# #             "delta": [utils.normalize_obj(o) for o in delta],
# #             "score": score,
# #             "diagnostics": diagnostics,
# #             "feedback_text": feedback_text,
# #         })
# #         with open(log_path, "w") as f:
# #             json.dump(self._refine_logs, f, indent=4)

# #         return score


# # class OrientationRefinerModule:
# #     """Replaces dspy.ChainOfThought(OrientationRefiner). (Unimplemented stub, as in original.)"""

# #     def __init__(self, save_dir):
# #         self.save_dir = save_dir

# #     def __call__(self, *args, **kwargs):
# #         return self.forward(*args, **kwargs)

# #     def forward(self, scene: Dict[str, Any], instruction) -> Dict[str, Any]:
# #         pass


# # class ScaleRefinerModule:
# #     """Replaces dspy.ChainOfThought(ScaleRefiner). (Unimplemented stub, as in original.)"""

# #     def __init__(self, save_dir):
# #         self.save_dir = save_dir

# #     def __call__(self, *args, **kwargs):
# #         return self.forward(*args, **kwargs)

# #     def forward(self, scene: Dict[str, Any]) -> Dict[str, Any]:
# #         pass


# # class ScenePlanner:
# #     """Top-level orchestrator (was a dspy.Module)."""

# #     def __init__(self, save_dir):
# #         self.scene_desc = SceneDescription()
# #         self.manager = ObjectPlacementManager(save_dir=save_dir)
# #         self.object_choice = ObjectChoice()
# #         self.planner = PlanAtomicTasks(save_dir=save_dir)
# #         self.orientation_refiner = OrientationRefinerModule(save_dir)
# #         self.scale_refiner = ScaleRefinerModule(save_dir)
# #         self.save_dir = save_dir

# #     def __call__(self, **kwargs):
# #         return self.forward(**kwargs)

# #     def forward(self, position, prev_scenegraphs, instruction, objects_list, dim_x, dim_y, dx, dy,
# #                 orientation=False, scale=False):

# #         env_note = f"Instruction: {instruction}" if instruction else "No special instruction."

# #         scene_desc, scene_name, scene_area = self.scene_desc(
# #             env_note=env_note,
# #             scene_desc=prev_scenegraphs[-1].get("scene_description", "") if len(prev_scenegraphs) != 0 else "No previous scene description!"
# #         )

# #         objects = self.object_choice.suggest_objects(env_note, scene_desc)

# #         plan, new_area_range, consistency_note, movement_note, allowed_objs = self.planner(
# #             position=position,
# #             env_note=env_note,
# #             scene_desc=scene_desc,
# #             object_store=objects_list,
# #             prev_scenegraphs=prev_scenegraphs,
# #             dim_x=dim_x, dim_y=dim_y, dx=dx, dy=dy,
# #         )

# #         scene = self.manager(position, plan, new_area_range, allowed_objs, dim_x, dim_y,
# #                              consistency_note, movement_note, base_scene={"objects": []}, env_note=env_note)
# #         utils.visualize(scene, position=pos, save_dir=save_dir, save_name="initial", sem_name="sem_plot")

# #         if orientation:
# #             scene_tmp = self.orientation_refiner(scene, instruction)
# #             scene["objects"] = [o.model_dump() for o in scene_tmp if o.label != "ego_person"]
# #             utils.visualize(scene, position=pos, save_dir=save_dir, save_name="orient")
# #         if scale:
# #             scene_tmp = self.scale_refiner(scene)
# #             scene["objects"] = [o.dict() for o in scene_tmp if o.label != "ego_person"]
# #             utils.visualize(scene, position=pos, save_dir=save_dir, save_name="scale")

# #         scene["Environmental_note"] = env_note
# #         scene["scene_description"] = scene_desc
# #         scene["Environment Name"] = scene_name
# #         scene["Scene Area"] = scene_area

# #         print(scene, "Scene")

# #         ego_obj = {
# #             "label": "ego_person",
# #             "object_type": "bulk",
# #             "anchor": None,
# #             "label_index": "ego_person",
# #             "position": [pos[0], pos[1]],
# #             "bbox": [pos[0], pos[1], 0, 0],
# #             "description": "Ego agent position",
# #             "rotation_angle": 0.0,
# #         }
# #         scene["objects"].append(ego_obj)

# #         with open(os.path.join(self.save_dir, f"scene_pos_{position[0]}_{position[1]}.json"), "w") as f:
# #             json.dump(scene, f, indent=2)

# #         return scene


# # if __name__ == "__main__":

# #     save_dir = input("Enter the path to save/load scenegraphs: ").strip()
# #     os.makedirs(save_dir, exist_ok=True)

# #     scene_planner = ScenePlanner(save_dir=save_dir)

# #     all_prev_scenes = []

# #     pos = (0, 0)

# #     prev_scene = {
# #         "objects": [],
# #         "scene_description": "No previous scene description.",
# #     }

# #     while True:
# #         instr = input("\nEnter environment instruction (or 'quit'): ").strip()
# #         if instr.lower() == "quit":
# #             break

# #         dx, dy = utils.extract_movement_from_note(instr)
# #         abs_x = pos[0] + dx * utils.STEP_SIZE
# #         abs_y = pos[1] + dy * utils.STEP_SIZE
# #         pos = (abs_x, abs_y)

# #         print("Ego moved to:", pos)

# #         scenegraph = utils.load_existing_scenegraph(pos, save_dir=save_dir)

# #         if scenegraph:
# #             print(f"Loaded existing scenegraph for position {pos}")
# #             result_scene = scenegraph
# #         else:
# #             obj_input = input("Enter a list of objects (comma-separated, e.g., 'chair, table, lamp'): ").strip()
# #             dim_x = int(input("x dimension of the room.").strip())
# #             dim_y = int(input("y dimension of the room.").strip())
# #             result_scene = scene_planner(
# #                 position=pos,
# #                 prev_scenegraphs=all_prev_scenes,
# #                 instruction=instr,
# #                 objects_list=obj_input,
# #                 dim_x=dim_x,
# #                 dim_y=dim_y,
# #                 dx=dx,
# #                 dy=dy,
# #                 orientation=False, scale=False,
# #             )

# #         utils.plot_scenegraph(result_scene, pos, save_dir=save_dir)
# #         print(f"Scene at position {pos} saved and plotted.")
# #         utils.visualize(result_scene, position=pos, save_dir=save_dir, save_name="final", sem_name="final_sem")
# #         prev_scene = result_scene
# #         all_prev_scenes.append(result_scene)


# """
# Two-step scene generation  (PURE PROMPT / NON-DSPY / NON-PYDANTIC VERSION)
# High-level Planning | Object placement at a go | Orientation | Scale

# #### WHOLE CONTEXT MANUALLY SET
# #### QWEN 72B
# #### WITHOUT COT, SCENE_DESC, ORNAMENTS
# #### WITH ORIENTATION, SCALE (prompt-driven, no DSPy)
# #### WITHOUT REFINE MODULE (replaced by a plain retry/scoring loop)
# #### WITHOUT TEXT, IMAGE CONDITIONING

# ##### BE CAREFUL ABOUT UPDATE INFORMATION - NO OPTION OF UPDATE HERE #####

# Notes on the port
# -----------------
# * No `dspy`, no `pydantic`. Every DSPy Signature became an explicit prompt +
#   a JSON-schema instruction block; every DSPy Module became a plain class.
# * Structured output is obtained by (a) telling the model the exact JSON shape,
#   and (b) parsing the reply with a string-aware balanced-brace extractor.
# * `dspy.Refine(N=5, reward_fn, threshold=1.0)` is reimplemented as an explicit
#   loop that re-prompts with feedback, scores each candidate, keeps the best,
#   and stops early once score >= threshold.
# * `dspy.ChainOfThought` is dropped; models are told to emit JSON only. The
#   parser tolerates any stray prose around the JSON, so light reasoning in the
#   reply won't break extraction.
# * SceneObject / AtomicTask are now plain dicts. All later `.model_dump()` /
#   `.dict()` / `.label` accesses were replaced with dict access.

# Fixed latent bugs from the DSPy version (were relying on __main__ globals):
# * `ObjectPlacementManager` referenced `dx`/`dy` that were never passed in -> now
#   threaded through `forward(...)`.
# * `placement_reward` referenced module-level `pos`/`save_dir` -> now uses the
#   position argument and `self.save_dir`.
# """

# import os
# import sys
# import re
# import io
# import json
# import base64
# import copy
# from typing import Dict, Any, List, Optional

# import numpy as np
# import matplotlib.pyplot as plt
# from matplotlib.patches import Rectangle
# from matplotlib.offsetbox import AnnotationBbox, OffsetImage
# from PIL import Image
# import networkx as nx

# import litellm

# # Add parent directory to Python path
# parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
# sys.path.append(parent_dir)
# sys.path.append("/home/kathakoli/3D-world-generation/multihop_approach")

# import plot_utils
# import utils_copy_img
# # from Evaluation import get_scene_evaluator
# # import render_blender   # only needed if orientation refinement is enabled


# # =============================================================================
# # LLM configuration
# # =============================================================================
# # Local vLLM (OpenAI-compatible) is the correct routing target. LiteLLM will
# # only hit the local server when ALL THREE hold: the "openai/" prefix,
# # a non-None api_base, and a non-empty api_key. Drop any of them and LiteLLM
# # silently falls back to the HuggingFace inference endpoint.
# MODEL       = "openai/Qwen/Qwen3.5-27B"
# API_BASE    = "http://localhost:7501/v1"
# API_KEY     = "local"
# TEMPERATURE = 0.5
# MAX_TOKENS  = 8000

# # HuggingFace fallback (uncomment to route to HF instead of local vLLM):
# # MODEL    = "huggingface/Qwen/Qwen2.5-VL-72B-Instruct"
# # API_BASE = None


# def llm_complete(prompt: str,
#                  images: Optional[List[str]] = None,
#                  system: Optional[str] = None,
#                  temperature: float = TEMPERATURE,
#                  max_tokens: int = MAX_TOKENS) -> str:
#     """
#     Single LiteLLM chat completion. `images` is a list of base64-encoded PNGs
#     (no data-URI prefix); when present the user turn becomes multimodal.
#     Returns the raw assistant text.
#     """
#     if images:
#         user_content: Any = [{"type": "text", "text": prompt}]
#         for b64 in images:
#             user_content.append({
#                 "type": "image_url",
#                 "image_url": {"url": f"data:image/png;base64,{b64}"},
#             })
#     else:
#         user_content = prompt

#     messages = []
#     if system:
#         messages.append({"role": "system", "content": system})
#     messages.append({"role": "user", "content": user_content})

#     kwargs = dict(
#         model=MODEL,
#         messages=messages,
#         api_key=API_KEY,
#         temperature=temperature,
#         max_tokens=max_tokens,
#     )
#     if API_BASE:
#         kwargs["api_base"] = API_BASE

#     resp = litellm.completion(**kwargs)
#     return resp.choices[0].message.content


# # =============================================================================
# # JSON extraction helpers (replace dspy.JSONAdapter + pydantic parsing)
# # =============================================================================
# def _strip_fences(text: str) -> str:
#     return re.sub(r"```(?:json)?", "", text, flags=re.I).strip()


# def _extract_balanced(text: str, open_ch: str, close_ch: str) -> Optional[str]:
#     """String-aware balanced-delimiter extractor (ignores braces inside strings)."""
#     start = text.find(open_ch)
#     if start == -1:
#         return None
#     depth = 0
#     in_str = False
#     esc = False
#     for i in range(start, len(text)):
#         c = text[i]
#         if in_str:
#             if esc:
#                 esc = False
#             elif c == "\\":
#                 esc = True
#             elif c == '"':
#                 in_str = False
#         else:
#             if c == '"':
#                 in_str = True
#             elif c == open_ch:
#                 depth += 1
#             elif c == close_ch:
#                 depth -= 1
#                 if depth == 0:
#                     return text[start:i + 1]
#     return None


# def parse_llm_json(raw: str) -> Any:
#     """
#     Parse the first valid JSON object OR array from an LM reply. Tolerates
#     markdown fences and surrounding prose. Raises ValueError if nothing parses.
#     """
#     text = _strip_fences(raw)

#     try:
#         return json.loads(text)
#     except Exception:
#         pass

#     # Prefer whichever delimiter appears first in the text.
#     first_obj = text.find("{")
#     first_arr = text.find("[")
#     order = []
#     if first_obj != -1:
#         order.append(("{", "}", first_obj))
#     if first_arr != -1:
#         order.append(("[", "]", first_arr))
#     order.sort(key=lambda t: t[2])

#     for open_ch, close_ch, _ in order:
#         candidate = _extract_balanced(text, open_ch, close_ch)
#         if candidate:
#             try:
#                 return json.loads(candidate)
#             except Exception:
#                 continue

#     raise ValueError(f"No valid JSON found in LM output:\n{raw[:2000]}")


# def coerce_obj_list(parsed: Any) -> List[Dict[str, Any]]:
#     """Pull a list of scene objects out of any of the shapes the model might emit."""
#     if isinstance(parsed, list):
#         return parsed
#     if isinstance(parsed, dict):
#         for key in ("objects", "placement_json", "updated_scene_json"):
#             val = parsed.get(key)
#             if isinstance(val, list):
#                 return val
#         scene = parsed.get("scene")
#         if isinstance(scene, dict) and isinstance(scene.get("objects"), list):
#             return scene["objects"]
#     return []


# def pil_to_b64(img: Image.Image) -> str:
#     buf = io.BytesIO()
#     img.convert("RGB").save(buf, format="PNG")
#     return base64.b64encode(buf.getvalue()).decode("utf-8")


# # =============================================================================
# # JSON schema instruction blocks (what the pydantic models used to enforce)
# # =============================================================================
# SCENE_DESC_SCHEMA = """
# Output ONLY a JSON object of exactly this shape (no markdown, no commentary):
# {
#   "new_scene_description": "<short, coherent description of the new scene>",
#   "concise_scene_name": "<one-word name for this area>",
#   "scene_area": "<indoor or outdoor>"
# }
# """.strip()

# OBJECT_CHOICE_SCHEMA = """
# Output ONLY a JSON object of exactly this shape (no markdown, no commentary):
# {
#   "bulk_objects": ["<large/structural/fixed object>", "..."],
#   "ornament_objects": ["<small/decorative/movable object>", "..."]
# }
# """.strip()

# SCENE_OBJECT_SCHEMA = """
# Output ONLY a JSON object of exactly this shape (no markdown, no commentary):
# {
#   "objects": [
#     {
#       "label": "<object name, same as plan, NO adjectives, singular>",
#       "object_type": "<'bulk' or 'ornaments'>",
#       "anchor": "<reference object label for a spatial relation, or null>",
#       "bbox": [x, y, w, h],
#       "position": [x + w/2, y + h/2],
#       "rotation_angle": <float degrees>,
#       "description": "<natural language description, may mention relations>",
#       "concise_description": "<description of ONLY this object, no relations>"
#     }
#   ]
# }
# Every object MUST include all of these fields.
# """.strip()

# ATOMIC_TASK_SCHEMA = """
# Output ONLY a JSON object of exactly this shape (no markdown, no commentary):
# {
#   "atomic_tasks": [
#     {
#       "id": "<unique task id>",
#       "object_type": "<'bulk' or 'ornaments'>",
#       "type": "<'Add' | 'Remove' | 'Update'>",
#       "label": "<object label this task refers to>",
#       "anchor": "<reference object for a spatial relation, or null>",
#       "spatial_hint": "<human-readable spatial instruction, or null>",
#       "bbox_hint": [x, y, w, h],
#       "output_schema": {
#         "add": [],
#         "remove": [],
#         "update": []
#       },
#       "execution_prompt": "<concise instruction to run this task>"
#     }
#   ]
# }
# 'bbox_hint' may be null. 'output_schema.add'/'update' hold full SceneObject dicts
# (label, object_type, anchor, bbox, position, rotation_angle, description,
# concise_description); 'remove' holds label strings. Leave them as empty lists
# when not needed.
# """.strip()


# # =============================================================================
# # Prompt Building  (unchanged from the DSPy version -- pure string logic)
# # =============================================================================
# def build_planning_prompt(position, env_note, prev_scenegraphs, save_dir, dim_x, dim_y,
#                           dx, dy, object_store, scene_desc=None, ornaments_included=False):
#     x, y = position
#     role_block = "Role: You are an expert layout planner.\n"

#     env_block = f"""
#     Environment note: {env_note}
#     """

#     placement_order = f"""
#     - Objects to be placed: {object_store}. Do not place any other objects than this list.
#     - DO NOT PLACE ANY ORNAMENTS.
#     - You have to place all the objects mentioned.
#     """

#     object_check = """
#     Strictly follow these rules:
#     - Objects not to include:
#         - windows, doors, walls, partitions or anything attached to it.
#         - floors, roofs.
#         - roads, street, pavement, footpath, sidewalk, pathway, lawn and any other part of ground.
#         - objects like counter that has multiple english meanings.
#     - Include only specific objects, not generalized names like new_object, appliances etc.
#     - Do not include any objects in plural.
#     """

#     prev_objects = []
#     for sg in prev_scenegraphs:
#         if isinstance(sg, dict):
#             if "scene" in sg and isinstance(sg["scene"], dict):
#                 prev_objects.extend(sg["scene"].get("objects", []))
#             else:
#                 prev_objects.extend(sg.get("objects", []))

#     prev_objects_summary = ", ".join([
#         f"{obj.get('label','unknown')} at {obj.get('position',[0,0])}"
#         for obj in prev_objects if obj.get("label") != "ego_person"
#     ])

#     if prev_objects:
#         if dy != 0:
#             dire = f"move along Y-axis by {dim_y} meters"
#             y = [position[1], position[1] + dim_y]
#             x = [position[0] - dim_x // 2, position[0] + dim_x // 2]
#             new_area_range = f"x ranging between {x[0]} meters and {x[1]} meters, y ranging between {y[0]} meters and {y[1]} meters"
#         elif dx != 0:
#             dire = f"move along X-axis by {dim_x} meters"
#             y = [position[1], position[1] + dim_y]
#             x = [position[0] - dim_x // 2, position[0] + dim_x // 2]
#             new_area_range = f"x ranging between {x[0]} meters and {x[1]} meters, y ranging between {y[0]} meters and {y[1]} meters"
#         else:
#             dire = "stay at same"
#             new_area_range = None

#         movement_block = f"""
#     Previous scene objects: {prev_objects_summary}
#     Movement Note: You {dire} to absolute ({position}).
#     """
#         consistency_block = """
#     Consistency Note:
#     - Objects mentioned in instruction should appear.
#     """
#     else:
#         x = [position[0] - dim_x // 2, position[0] + dim_x // 2]
#         y = [position[1], position[1] + dim_y]
#         new_area_range = f"x ranging between {x[0]} meters and {x[1]} meters, y ranging between {y[0]} meters and {y[1]} meters"
#         movement_block = f"Movement Note: You are at {position}."
#         consistency_block = f"""
#     Consistency Note:
#     - Initialize scene with new objects ONLY in the visible area: {new_area_range}.
#     - Generate a scene description.
#     """

#     planning_prompt = role_block + env_block + placement_order + object_check

#     if scene_desc:
#         scene_desc_block = f"""
#         New Scene Description:
#         {scene_desc}
#         """
#         planning_prompt = planning_prompt + scene_desc_block

#     semantic_block = """
#     Object Arrangement and Spatial Relationships:
#     - Objects should be arranged naturally as seen in real world environment (e.g., chairs around table, cars parked along street, trees spaced in a park, lamp on top of table/desk, pillow on sofa/bed , television facing sofa , vase centered on table ,  bird feeders hanging on a tree, trash bins beside bench ).
#     - Define clear spatial relationships between objects (e.g., next to, under, near, on top of, centered on, beside).
#     - Align most larger objects along area boundaries(e.g. walls, fences or paths) and corners to reflect natural real-world arrangements and maximise navigable spaces.
#     - Ensure bounding boxes and positions are consistent with the defined spatial relationships.

#     Follow semantic asset grouping:
#     - A semantic asset group is a collection of assets that are logically, functionally, or spatially related within a scene. Assets in a group support a common purpose or activity and together form a meaningful subscene.
#     - Group assets based on function (used together), semantics (conceptually related), or geometry (close in space or aligned).
#     - Examples:
#         - Sleeping area: Bed + Nightstand + Lamp (used together for resting and reading)
#         - Entertainment area: Sofa + TV Console + TV  (functionally related for watching)
#         - Work area: Desk + Chair + Laptop (supports the working activity)
#         - Dining area: Dining Table + Chairs + Chandelier (forms a cohesive eating zone)
#     """

#     task_block = """
# Task:
# Decompose this instruction into ATOMIC TASKS for the design process. The output should ONLY be a valid json."""
#     planning_prompt = planning_prompt + semantic_block + task_block

#     file_path = os.path.join(save_dir, f"planning_prompt_{position[0]}_{position[1]}.txt")
#     with open(file_path, "w", encoding="utf-8") as f:
#         f.write(planning_prompt)

#     return planning_prompt, new_area_range, consistency_block, movement_block


# def build_object_placement_prompt(plan_text, position, dim_x, dim_y, consistency_block,
#                                   movement_block, env_note, new_area_range):
#     role_block = "Role: You are an expert real world planner.\n"
#     env_block = f"Environment note: {env_note}.\n"
#     return role_block + env_block + f"""

# Plan summary for the room:
# {plan_text}

# Placement Rules:
# - You are populating a {dim_x} meters * {dim_y} meters room.
# - The room dimensions are x ranging between -{dim_x/2} to {dim_x/2} and y ranging between 0 to {dim_y}. Bounding boxes SHOULD NOT go out of the room area.
# - Design such that two BULK objects bounding boxes SHOULD NOT INTERSECT each other unless the relationship is under or on top of.
# - While placing, verify that the front of the object being placed and already placed objects remains clearly visible and is not obstructed by any larger object positioned in front of it.
# - Maintain realistic distances between the objects you are placing to make the room visually appealing. Do not clutter them together.
# - Align most larger objects along area boundaries(e.g. walls, fences or paths) to reflect natural real-world arrangements and maximise navigable spaces.
# - Each object must include: label(same as the name mentioned in plan, do not add any adjective), object_type, anchor, description, bbox [x,y,w,h] where (x+w/2,y+h/2) is the position/center, (x,y) is the bottom left corner and (w,h) is the width and height, description, position [x+w/2,y+h/2], rotation angle.
# - Object sizes and bounding boxes must be semantically consistent with other objects in the scene and the room dimension.
# - Positions and bounding boxes must accurately reflect spatial relationships (e.g., next to, under, on top of, centered on, near).
# - Anchored objects must be placed and oriented relative to their reference object according to the defined spatial relationship.

# Instructions:
# - Output ONLY valid JSON with fields: objects.
# """


# def build_orientation_prompt(scene: Dict[str, Any]) -> str:
#     objs = scene.get("objects", [])
#     obj_lines = "\n".join([
#         f"- {o['label']}: {o.get('description', '')} (anchor: {o.get('anchor', 'none')})"
#         for o in objs
#     ])
#     return f"""
# You are refining object orientations in a 3D indoor or outdoor scene.
# Assign realistic rotation angles (in degrees, 0-360) to each object.

# Interpretation:
# - 0 deg: facing forward (toward the ego or camera)
# - 90 deg: facing right
# - 180 deg: facing backward
# - 270 deg: facing left

# Guidelines:
# - Objects attached to walls should face away from the wall.
# - Sofas and chairs should face TVs or tables if present.
# - Tables should align with nearby seating.
# - TVs and monitors should face the main sitting area.
# - Beds should face away from their headboard wall.
# - Cabinets, wardrobes, and shelves should be against walls and face outward.
# - Appliances (stoves, sinks, fridges) should face the open workspace or usable area.
# - Floor lamps, indoor plants, and static decor items usually have 0 deg rotation.
# - Benches should face open spaces, walkways, or scenic views.
# - Vehicles should face their direction of travel or toward open space.
# - Trees, rocks, and natural outdoor elements should maintain 0 deg rotation.
# - Freestanding objects in open areas should face the scene center or main viewpoint.
# - Groups of similar objects (e.g., multiple chairs) should be aligned symmetrically.
# - Desks should face open space or toward the user's seating direction.
# - Bathroom fixtures (sinks, toilets) should face the center or entry of the room, not walls.

# Scene objects:
# {scene}

# - Output ONLY valid JSON with fields: objects exactly similar to input. Do not output reasoning.
# """.strip()


# def build_scale_prompt(scene: Dict[str, Any]) -> str:
#     print(scene, "Scene build")
#     objs = scene.get("objects", [])
#     obj_labels = ", ".join([o["label"] for o in objs])
#     return f"""
# You are refining object scales in a realistic 3D scene.

# Adjust the size of each object's bounding box based on its natural proportion relative to surrounding objects in the scene. Maintain realistic scale relationships as seen in real environments.
# Ensure relative size consistency - smaller items (e.g., lamps, books, vases) should not exceed larger reference objects (e.g., tables, beds, sofas).
# Examples:
# - A chair should be smaller than a desk/table.
# - A desk should be smaller than a bed.
# - A nightstand should be lower than a bed.
# - A sofa should be larger than a side table.
# - Trees should be taller than nearby benches or cars.
# - Buildings should dwarf trees and vehicles.
# Keep proportions coherent within the environment (indoor or outdoor), adjusting bounding box dimensions for necessary objects so all objects appear physically plausible when viewed together.

# Scene objects:
# {scene}

# - Output ONLY valid JSON with fields: objects exactly similar to input.
# """.strip()


# # =============================================================================
# # Modules  (plain classes -- no dspy.Module, no ChainOfThought)
# # =============================================================================
# class SceneDescription:
#     """Replaces SceneDescriptionGenerator signature + SceneDescription module."""

#     def __call__(self, env_note: str, scene_desc: str):
#         print("\n Generating new scene description for position ")
#         seed = utils_copy_img.generate_random_seed()
#         prompt = f"""Role: You generate a concise updated description of the current scene using prior context, environment notes, and predicted area type.

# Random seed: {seed}
# Previous scene description: {scene_desc}
# Environment note: {env_note}

# {SCENE_DESC_SCHEMA}"""
#         raw = llm_complete(prompt)
#         try:
#             data = parse_llm_json(raw)
#         except ValueError:
#             data = {}
#         new_scene_desc = data.get("new_scene_description", "No scene generated.")
#         new_scene_name = data.get("concise_scene_name", "No scene generated.")
#         new_scene_area = data.get("scene_area", "No scene generated.")
#         return new_scene_desc, new_scene_name, new_scene_area


# class ObjectChoice:
#     """Replaces ObjectChoicePlanner signature + ObjectChoice module."""

#     def suggest_objects(self, env_note: str, scene_description: str) -> Dict[str, List[str]]:
#         seed = utils_copy_img.generate_random_seed()
#         choice_restriction_prompt = """
#             Strictly follow these rules:
# - Objects not to include:
#     - windows, doors, walls, partitions or anything attached to it.
#     - floors, roofs.
#     - roads, street, pavement, footpath, sidewalk, pathway,lawn and any other part of ground.
#     - objects like counter that has multiple english meanings.
# - Include only specific objects, not generalized names like new_object, appliances etc.
# - Do not include any objects in plural.
# - Do not repeat any object more than twice. Try to avoid repeating objects.
#         """
#         prompt = f"""Role: Decide potential bulk and ornament objects for the described scene.

# Random seed: {seed}
# Environment note: {env_note}
# Scene description: {scene_description}
# Choice restrictions: {choice_restriction_prompt}

# bulk_objects: large, structural, or fixed objects (e.g., building, tree, table, sofa, road).
# ornament_objects: smaller, decorative, or movable objects (e.g., lamp, plant, trash bin, signboard).

# {OBJECT_CHOICE_SCHEMA}"""
#         raw = llm_complete(prompt)
#         try:
#             data = parse_llm_json(raw)
#         except ValueError:
#             data = {}
#         bulk = data.get("bulk_objects") or []
#         ornaments = data.get("ornament_objects") or []
#         return {"bulk_objects": bulk, "ornament_objects": ornaments}


# class PlanAtomicTasks:
#     """Replaces AtomicTaskPlanning signature + PlanAtomicTasks module."""

#     def __init__(self, save_dir):
#         self.save_dir = save_dir

#     def forward(self, position, env_note, scene_desc, object_store, prev_scenegraphs,
#                 dim_x, dim_y, dx, dy):
#         planning_prompt, new_area_range, consistency_note, movement_note = build_planning_prompt(
#             position=position, env_note=env_note, prev_scenegraphs=prev_scenegraphs,
#             save_dir=self.save_dir, dim_x=dim_x, dim_y=dim_y, dx=dx, dy=dy,
#             object_store=object_store, scene_desc=scene_desc, ornaments_included=False,
#         )

#         allowed_objs = utils_copy_img._parse_allowed_objects(planning_prompt)
#         print("Inside planner")

#         seed = utils_copy_img.generate_random_seed()
#         full_prompt = f"""Random seed: {seed}

# {planning_prompt}

# {ATOMIC_TASK_SCHEMA}"""
#         raw = llm_complete(full_prompt)
#         print("After planning computation")

#         parsed = parse_llm_json(raw)
#         # Normalize to {"atomic_tasks": [...]}
#         if isinstance(parsed, list):
#             parsed_pred = {"atomic_tasks": parsed}
#         elif isinstance(parsed, dict) and "atomic_tasks" in parsed:
#             parsed_pred = {"atomic_tasks": parsed["atomic_tasks"]}
#         else:
#             parsed_pred = {"atomic_tasks": coerce_obj_list(parsed)}

#         with open(os.path.join(self.save_dir, f"planning_{position[0]}_{position[1]}.json"), "w") as f:
#             json.dump(parsed_pred, f, indent=2)
#         return parsed_pred, new_area_range, consistency_note, movement_note, allowed_objs


# class ObjectPlacementManager:
#     """
#     Replaces ObjectPlacementGenerator signature + dspy.Refine.
#     Manual best-of-N refinement loop with feedback re-prompting.
#     """

#     def __init__(self, save_dir, N=5, threshold=1.0):
#         self.save_dir = save_dir
#         self.N = N
#         self.threshold = threshold
#         self._refine_logs = []

#     def _generate_placement(self, instruction_prompt, feedback, seed):
#         parts = [f"Random seed: {seed}", instruction_prompt]
#         if feedback:
#             parts.append("Feedback based on previous placements. Follow this feedback:\n" + feedback)
#         parts.append(SCENE_OBJECT_SCHEMA)
#         prompt = "\n\n".join(parts)
#         raw = llm_complete(prompt)
#         parsed = parse_llm_json(raw)
#         return coerce_obj_list(parsed)

#     def forward(self, position, plan, new_area_range, allowed_objs, dim_x, dim_y,
#                 consistency_block, movement_block, base_scene, dx, dy, env_note="No note"):
#         plan_text = "\n".join(
#             task.get('execution_prompt', '')
#             for task in plan.get("atomic_tasks", [])
#             if task.get('execution_prompt')
#         )
#         instruction_prompt = build_object_placement_prompt(
#             plan_text, position, dim_x, dim_y, consistency_block, movement_block,
#             env_note, new_area_range,
#         )

#         new_area_range_parsed = utils_copy_img._parse_new_area_range(new_area_range)

#         best_score = float("-inf")
#         best_delta: List[Dict[str, Any]] = []
#         feedback = ""
#         self._refine_logs = []

#         for i in range(self.N):
#             seed = utils_copy_img.generate_random_seed()
#             delta = self._generate_placement(instruction_prompt, feedback, seed)

#             # --- visualize this candidate ---
#             tmp_scene = copy.deepcopy(base_scene)
#             delta_norm = [utils_copy_img.normalize_obj(o) for o in delta]
#             tmp_scene["objects"].extend(delta_norm)
#             utils_copy_img.visualize(
#                 tmp_scene, position=position, save_dir=self.save_dir,
#                 save_name=f"check_{i + 1}",
#             )

#             # --- score candidate ---
#             score, diagnostics = utils_copy_img.score_candidate_delta(
#                 delta, base_scene, allowed_objs, new_area_range_parsed, position, dx, dy,
#             )

#             feedback = (
#                 "Placement Diagnostics / Issues:\n"
#                 + "\n".join(f"- {d}" for d in diagnostics)
#                 + "\n\nAction Required:\n"
#                 "Update the positions or bounding box dimensions of the objects having "
#                 "placement problems to address the above issues. Ensure that any changes "
#                 "do not cause bounding box overlaps with other objects in the scene."
#             )

#             self._refine_logs.append({
#                 "iteration": i + 1,
#                 "delta": delta_norm,
#                 "score": score,
#                 "diagnostics": diagnostics,
#                 "feedback_text": feedback,
#             })
#             log_path = os.path.join(self.save_dir, f"refine_logs__{position[0]}_{position[1]}.json")
#             with open(log_path, "w") as f:
#                 json.dump(self._refine_logs, f, indent=4)

#             if score > best_score:
#                 best_score = score
#                 best_delta = delta

#             if score >= self.threshold:
#                 break

#         # --- commit the best candidate ---
#         delta_objects = [utils_copy_img.normalize_obj(o) for o in best_delta]
#         if delta_objects:
#             delta_objects = utils_copy_img.assign_progressive_ids(
#                 delta_objects, scene_graph_dir=self.save_dir,
#             )
#         base_scene["objects"].extend(delta_objects)
#         return base_scene


# class OrientationRefinerModule:
#     """Replaces OrientationRefiner signature + OrientationRefinerModule (multimodal)."""

#     def __init__(self, save_dir):
#         self.save_dir = save_dir

#     def forward(self, scene: Dict[str, Any], instruction) -> List[Dict[str, Any]]:
#         print("\n[OrientationRefiner] Refining object rotations...")
#         updated_scene = copy.deepcopy(scene)
#         objects = updated_scene.get("objects", [])
#         if not objects:
#             print("No objects found for orientation refinement.")
#             return objects

#         for obj in scene.get("objects", []):
#             if "rotation_angle" not in obj or obj["rotation_angle"] is None:
#                 obj["rotation_angle"] = 0.0

#         prompt = build_orientation_prompt({"objects": objects})

#         json_path = os.path.join(self.save_dir, "check_tmp.json")
#         with open(json_path, "w") as f:
#             json.dump(scene, f, indent=2)
#         print(f"[Saved delta JSON] {json_path}")

#         blender_dir = os.path.join(self.save_dir, "check_tmp")
#         # Requires `import render_blender` at top to be uncommented.
#         render_blender.run_blender(
#             json_path=json_path,
#             glb_directory="asset_library",
#             output_directory=blender_dir,
#             environment_name=instruction,
#         )

#         images = [
#             pil_to_b64(Image.open(os.path.join(blender_dir, "view_00.png"))),  # side 1
#             pil_to_b64(Image.open(os.path.join(blender_dir, "view_02.png"))),  # side 2
#             pil_to_b64(Image.open(os.path.join(blender_dir, "view_01.png"))),  # front
#             pil_to_b64(Image.open(os.path.join(blender_dir, "view_03.png"))),  # back
#             pil_to_b64(Image.open(os.path.join(blender_dir, "view_04.png"))),  # top
#         ]

#         seed = utils_copy_img.generate_random_seed()
#         full_prompt = f"""Random seed: {seed}

# {prompt}

# Current placement JSON:
# {json.dumps(objects, indent=2)}

# The five attached images are: side view 1, side view 2, front view, back view, top view.
# Refer to them to critique and correct each object's rotation_angle.
# Return the SAME list of objects with ONLY rotation_angle updated.

# {SCENE_OBJECT_SCHEMA}"""

#         raw = llm_complete(full_prompt, images=images)
#         parsed = parse_llm_json(raw)
#         return coerce_obj_list(parsed)


# class ScaleRefinerModule:
#     """Replaces ScaleRefiner signature + ScaleRefinerModule."""

#     def __init__(self, save_dir):
#         self.save_dir = save_dir

#     def forward(self, scene: Dict[str, Any]) -> List[Dict[str, Any]]:
#         print("\n[ScaleRefiner] Refining object scales...")
#         updated_scene = copy.deepcopy(scene)
#         objects = updated_scene.get("objects", [])
#         if not objects:
#             print("No objects found for scale refinement.")
#             return objects

#         prompt = build_scale_prompt({"objects": objects})
#         seed = utils_copy_img.generate_random_seed()
#         full_prompt = f"""Random seed: {seed}

# {prompt}

# Current placement JSON:
# {json.dumps(objects, indent=2)}

# Return the SAME list of objects with ONLY bbox and position updated.

# {SCENE_OBJECT_SCHEMA}"""

#         raw = llm_complete(full_prompt)
#         parsed = parse_llm_json(raw)
#         return coerce_obj_list(parsed)


# # =============================================================================
# # Orchestrator
# # =============================================================================
# class ScenePlanner:
#     def __init__(self, save_dir):
#         self.scene_desc = SceneDescription()
#         self.manager = ObjectPlacementManager(save_dir=save_dir)
#         self.object_choice = ObjectChoice()
#         self.planner = PlanAtomicTasks(save_dir=save_dir)
#         self.orientation_refiner = OrientationRefinerModule(save_dir)
#         self.scale_refiner = ScaleRefinerModule(save_dir)
#         self.save_dir = save_dir

#     def forward(self, position, prev_scenegraphs, instruction, objects_list, dim_x, dim_y,
#                 dx, dy, orientation=False, scale=False):

#         env_note = f"Instruction: {instruction}" if instruction else "No special instruction."

#         scene_desc, scene_name, scene_area = self.scene_desc(
#             env_note=env_note,
#             scene_desc=prev_scenegraphs[-1].get("scene_description", "")
#             if len(prev_scenegraphs) != 0 else "No previous scene description!",
#         )

#         objects = self.object_choice.suggest_objects(env_note, scene_desc)

#         plan, new_area_range, consistency_note, movement_note, allowed_objs = self.planner.forward(
#             position=position, env_note=env_note, scene_desc=scene_desc,
#             object_store=objects_list, prev_scenegraphs=prev_scenegraphs,
#             dim_x=dim_x, dim_y=dim_y, dx=dx, dy=dy,
#         )

#         scene = self.manager.forward(
#             position, plan, new_area_range, allowed_objs, dim_x, dim_y,
#             consistency_note, movement_note, base_scene={"objects": []},
#             dx=dx, dy=dy, env_note=env_note,
#         )
#         utils_copy_img.visualize(scene, position=position, save_dir=self.save_dir, save_name="initial")

#         # Refinement passes (objects are plain dicts now)
#         if orientation:
#             scene_tmp = self.orientation_refiner.forward(scene, instruction)
#             scene["objects"] = [o for o in scene_tmp if o.get("label") != "ego_person"]
#             utils_copy_img.visualize(scene, position=position, save_dir=self.save_dir, save_name="orient")
#         if scale:
#             scene_tmp = self.scale_refiner.forward(scene)
#             scene["objects"] = [o for o in scene_tmp if o.get("label") != "ego_person"]
#             utils_copy_img.visualize(scene, position=position, save_dir=self.save_dir, save_name="scale")

#         scene["Environmental_note"] = env_note
#         scene["scene_description"] = scene_desc
#         scene["Environment Name"] = scene_name
#         scene["Scene Area"] = scene_area

#         print(scene, "Scene")

#         ego_obj = {
#             "label": "ego_person",
#             "object_type": "bulk",
#             "anchor": None,
#             "label_index": "ego_person",
#             "position": [position[0], position[1]],
#             "bbox": [position[0], position[1], 0, 0],
#             "description": "Ego agent position",
#             "rotation_angle": 0.0,
#         }
#         scene["objects"].append(ego_obj)

#         with open(os.path.join(self.save_dir, f"scene_pos_{position[0]}_{position[1]}.json"), "w") as f:
#             json.dump(scene, f, indent=2)

#         return scene


# # =============================================================================
# # Main
# # =============================================================================
# if __name__ == "__main__":

#     save_dir = input("Enter the path to save/load scenegraphs: ").strip()
#     os.makedirs(save_dir, exist_ok=True)

#     scene_planner = ScenePlanner(save_dir=save_dir)

#     all_prev_scenes = []
#     pos = (0, 0)

#     prev_scene = {
#         "objects": [],
#         "scene_description": "No previous scene description.",
#     }

#     while True:
#         instr = input("\nEnter environment instruction (or 'quit'): ").strip()
#         if instr.lower() == "quit":
#             break

#         dx, dy = utils_copy_img.extract_movement_from_note(instr)
#         abs_x = pos[0] + dx * utils_copy_img.STEP_SIZE
#         abs_y = pos[1] + dy * utils_copy_img.STEP_SIZE
#         pos = (abs_x, abs_y)
#         print("Ego moved to:", pos)

#         scenegraph = utils_copy_img.load_existing_scenegraph(pos, save_dir=save_dir)

#         if scenegraph:
#             print(f"Loaded existing scenegraph for position {pos}")
#             result_scene = scenegraph
#         else:
#             obj_input = input("Enter a list of objects (comma-separated, e.g., 'chair, table, lamp'): ").strip()
#             dim_x = int(input("x dimension of the room.").strip())
#             dim_y = int(input("y dimension of the room.").strip())
#             result_scene = scene_planner.forward(
#                 position=pos,
#                 prev_scenegraphs=all_prev_scenes,
#                 instruction=instr,
#                 objects_list=obj_input,
#                 dim_x=dim_x,
#                 dim_y=dim_y,
#                 dx=dx,
#                 dy=dy,
#                 orientation=False, scale=False,
#             )

#         utils_copy_img.plot_scenegraph(result_scene, pos, save_dir=save_dir)
#         print(f"Scene at position {pos} saved and plotted.")
#         utils_copy_img.visualize(result_scene, position=pos, save_dir=save_dir, save_name="final")
#         prev_scene = result_scene
#         all_prev_scenes.append(result_scene)

"""
Two-step scene generation  (PURE PROMPT / NON-DSPY / NON-PYDANTIC VERSION)
High-level Planning | Object placement at a go | Orientation | Scale

#### WHOLE CONTEXT MANUALLY SET
#### QWEN 72B
#### WITHOUT COT, SCENE_DESC, ORNAMENTS
#### WITH ORIENTATION, SCALE (prompt-driven, no DSPy)
#### WITHOUT REFINE MODULE (replaced by a plain retry/scoring loop)
#### WITHOUT TEXT, IMAGE CONDITIONING

##### BE CAREFUL ABOUT UPDATE INFORMATION - NO OPTION OF UPDATE HERE #####

Notes on the port
-----------------
* No `dspy`, no `pydantic`. Every DSPy Signature became an explicit prompt +
  a JSON-schema instruction block; every DSPy Module became a plain class.
* Structured output is obtained by (a) forcing vLLM JSON mode via
  response_format={"type":"json_object"}, (b) a hard JSON-only system prompt,
  and (c) a string-aware balanced-brace parser with a last-ditch fallback.
* `dspy.Refine(N=5, reward_fn, threshold=1.0)` is reimplemented as an explicit
  loop that re-prompts with feedback, scores each candidate, keeps the best,
  and stops early once score >= threshold.
* `dspy.ChainOfThought` is dropped; models are told to emit JSON only. JSON mode
  prevents "Thinking Process:" preambles from ever appearing.
* SceneObject / AtomicTask are now plain dicts. All later `.model_dump()` /
  `.dict()` / `.label` accesses were replaced with dict access.

Fixed latent bugs from the DSPy version (were relying on __main__ globals):
* `ObjectPlacementManager` referenced `dx`/`dy` that were never passed in -> now
  threaded through `forward(...)`.
* `placement_reward` referenced module-level `pos`/`save_dir` -> now uses the
  position argument and `self.save_dir`.
"""

import os
import sys
import re
import io
import json
import base64
import copy
from typing import Dict, Any, List, Optional

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
from PIL import Image
import networkx as nx

import litellm

# Add parent directory to Python path
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)
sys.path.append("/home/kathakoli/3D-world-generation/multihop_approach")

import plot_utils
import utils_copy_img
# from Evaluation import get_scene_evaluator
# import render_blender   # only needed if orientation refinement is enabled


# =============================================================================
# LLM configuration
# =============================================================================
# Local vLLM (OpenAI-compatible) is the correct routing target. LiteLLM will
# only hit the local server when ALL THREE hold: the "openai/" prefix,
# a non-None api_base, and a non-empty api_key. Drop any of them and LiteLLM
# silently falls back to the HuggingFace inference endpoint.
MODEL       = "openai/Qwen/Qwen3.5-27B"
API_BASE    = "http://localhost:7501/v1"
API_KEY     = "local"
TEMPERATURE = 0.5
MAX_TOKENS  = 12000   # generous so a valid JSON object can't get truncated mid-object

# HuggingFace fallback (uncomment to route to HF instead of local vLLM):
# MODEL    = "huggingface/Qwen/Qwen2.5-VL-72B-Instruct"
# API_BASE = None

JSON_SYSTEM = (
    "You are a strict JSON generator. Output ONLY a single valid JSON value "
    "and nothing else. Do NOT include any reasoning, thinking, explanation, "
    "preamble, markdown, or code fences. Your entire response must be parseable "
    "by json.loads()."
)


def llm_complete(prompt: str,
                 images: Optional[List[str]] = None,
                 system: Optional[str] = None,
                 temperature: float = TEMPERATURE,
                 max_tokens: int = MAX_TOKENS,
                 force_json: bool = True) -> str:
    """
    Single LiteLLM chat completion. `images` is a list of base64-encoded PNGs
    (no data-URI prefix); when present the user turn becomes multimodal.

    When force_json is True:
      * a hard JSON-only system prompt is injected (unless caller overrides), and
      * response_format={"type":"json_object"} asks vLLM to constrain generation
        to valid JSON via guided decoding. If the server rejects that param
        (no guided-decoding backend), we retry once without it and lean on the
        system prompt alone.

    Returns the raw assistant text.
    """
    if images:
        user_content: Any = [{"type": "text", "text": prompt}]
        for b64 in images:
            user_content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{b64}"},
            })
    else:
        user_content = prompt

    messages = []
    sys_msg = system or (JSON_SYSTEM if force_json else None)
    if sys_msg:
        messages.append({"role": "system", "content": sys_msg})
    messages.append({"role": "user", "content": user_content})

    kwargs = dict(
        model=MODEL,
        messages=messages,
        api_key=API_KEY,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    if API_BASE:
        kwargs["api_base"] = API_BASE
    if force_json:
        kwargs["response_format"] = {"type": "json_object"}

    try:
        resp = litellm.completion(**kwargs)
    except Exception:
        # Server may reject response_format if no guided-decoding backend is up.
        if "response_format" in kwargs:
            kwargs.pop("response_format")
            resp = litellm.completion(**kwargs)
        else:
            raise
    return resp.choices[0].message.content


# =============================================================================
# JSON extraction helpers (replace dspy.JSONAdapter + pydantic parsing)
# =============================================================================
def _strip_fences(text: str) -> str:
    return re.sub(r"```(?:json)?", "", text, flags=re.I).strip()


def _extract_balanced(text: str, open_ch: str, close_ch: str) -> Optional[str]:
    """String-aware balanced-delimiter extractor (ignores braces inside strings)."""
    start = text.find(open_ch)
    if start == -1:
        return None
    depth = 0
    in_str = False
    esc = False
    for i in range(start, len(text)):
        c = text[i]
        if in_str:
            if esc:
                esc = False
            elif c == "\\":
                esc = True
            elif c == '"':
                in_str = False
        else:
            if c == '"':
                in_str = True
            elif c == open_ch:
                depth += 1
            elif c == close_ch:
                depth -= 1
                if depth == 0:
                    return text[start:i + 1]
    return None


def parse_llm_json(raw: str) -> Any:
    """
    Parse the first valid JSON object OR array from an LM reply. Tolerates
    markdown fences and surrounding prose. Raises ValueError if nothing parses.
    """
    text = _strip_fences(raw)

    try:
        return json.loads(text)
    except Exception:
        pass

    # Prefer whichever delimiter appears first in the text.
    first_obj = text.find("{")
    first_arr = text.find("[")
    order = []
    if first_obj != -1:
        order.append(("{", "}", first_obj))
    if first_arr != -1:
        order.append(("[", "]", first_arr))
    order.sort(key=lambda t: t[2])

    for open_ch, close_ch, _ in order:
        candidate = _extract_balanced(text, open_ch, close_ch)
        if candidate:
            try:
                return json.loads(candidate)
            except Exception:
                continue

    # last-ditch: try the last {...} block in case prose contains stray braces
    last = text.rfind("{")
    if last != -1:
        cand = _extract_balanced(text[last:], "{", "}")
        if cand:
            try:
                return json.loads(cand)
            except Exception:
                pass

    raise ValueError(f"No valid JSON found in LM output:\n{raw[:2000]}")


def coerce_obj_list(parsed: Any) -> List[Dict[str, Any]]:
    """Pull a list of scene objects out of any of the shapes the model might emit."""
    if isinstance(parsed, list):
        return parsed
    if isinstance(parsed, dict):
        for key in ("objects", "placement_json", "updated_scene_json"):
            val = parsed.get(key)
            if isinstance(val, list):
                return val
        scene = parsed.get("scene")
        if isinstance(scene, dict) and isinstance(scene.get("objects"), list):
            return scene["objects"]
    return []


def pil_to_b64(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


# =============================================================================
# JSON schema instruction blocks (what the pydantic models used to enforce)
# =============================================================================
SCENE_DESC_SCHEMA = """
Output ONLY a JSON object of exactly this shape (no markdown, no commentary):
{
  "new_scene_description": "<short, coherent description of the new scene>",
  "concise_scene_name": "<one-word name for this area>",
  "scene_area": "<indoor or outdoor>"
}
""".strip()

OBJECT_CHOICE_SCHEMA = """
Output ONLY a JSON object of exactly this shape (no markdown, no commentary):
{
  "bulk_objects": ["<large/structural/fixed object>", "..."],
  "ornament_objects": ["<small/decorative/movable object>", "..."]
}
""".strip()

SCENE_OBJECT_SCHEMA = """
Output ONLY a JSON object of exactly this shape (no markdown, no commentary):
{
  "objects": [
    {
      "label": "<object name, same as plan, NO adjectives, singular>",
      "object_type": "<'bulk' or 'ornaments'>",
      "anchor": "<reference object label for a spatial relation, or null>",
      "bbox": [x, y, w, h],
      "position": [x + w/2, y + h/2],
      "rotation_angle": <float degrees>,
      "description": "<natural language description, may mention relations>",
      "concise_description": "<description of ONLY this object, no relations>"
    }
  ]
}
Every object MUST include all of these fields.
""".strip()

ATOMIC_TASK_SCHEMA = """
Output ONLY a JSON object of exactly this shape (no markdown, no commentary):
{
  "atomic_tasks": [
    {
      "id": "<unique task id>",
      "object_type": "<'bulk' or 'ornaments'>",
      "type": "<'Add' | 'Remove' | 'Update'>",
      "label": "<object label this task refers to>",
      "anchor": "<reference object for a spatial relation, or null>",
      "spatial_hint": "<human-readable spatial instruction, or null>",
      "bbox_hint": [x, y, w, h],
      "output_schema": {
        "add": [],
        "remove": [],
        "update": []
      },
      "execution_prompt": "<concise instruction to run this task>"
    }
  ]
}
'bbox_hint' may be null. 'output_schema.add'/'update' hold full SceneObject dicts
(label, object_type, anchor, bbox, position, rotation_angle, description,
concise_description); 'remove' holds label strings. Leave them as empty lists
when not needed.
""".strip()


# =============================================================================
# Prompt Building  (unchanged from the DSPy version -- pure string logic)
# =============================================================================
def build_planning_prompt(position, env_note, prev_scenegraphs, save_dir, dim_x, dim_y,
                          dx, dy, object_store, scene_desc=None, ornaments_included=False):
    x, y = position
    role_block = "Role: You are an expert layout planner.\n"

    env_block = f"""
    Environment note: {env_note}
    """

    placement_order = f"""
    - Objects to be placed: {object_store}. Do not place any other objects than this list.
    - DO NOT PLACE ANY ORNAMENTS.
    - You have to place all the objects mentioned.
    """

    object_check = """
    Strictly follow these rules:
    - Objects not to include:
        - windows, doors, walls, partitions or anything attached to it.
        - floors, roofs.
        - roads, street, pavement, footpath, sidewalk, pathway, lawn and any other part of ground.
        - objects like counter that has multiple english meanings.
    - Include only specific objects, not generalized names like new_object, appliances etc.
    - Do not include any objects in plural.
    """

    prev_objects = []
    for sg in prev_scenegraphs:
        if isinstance(sg, dict):
            if "scene" in sg and isinstance(sg["scene"], dict):
                prev_objects.extend(sg["scene"].get("objects", []))
            else:
                prev_objects.extend(sg.get("objects", []))

    prev_objects_summary = ", ".join([
        f"{obj.get('label','unknown')} at {obj.get('position',[0,0])}"
        for obj in prev_objects if obj.get("label") != "ego_person"
    ])

    if prev_objects:
        if dy != 0:
            dire = f"move along Y-axis by {dim_y} meters"
            y = [position[1], position[1] + dim_y]
            x = [position[0] - dim_x // 2, position[0] + dim_x // 2]
            new_area_range = f"x ranging between {x[0]} meters and {x[1]} meters, y ranging between {y[0]} meters and {y[1]} meters"
        elif dx != 0:
            dire = f"move along X-axis by {dim_x} meters"
            y = [position[1], position[1] + dim_y]
            x = [position[0] - dim_x // 2, position[0] + dim_x // 2]
            new_area_range = f"x ranging between {x[0]} meters and {x[1]} meters, y ranging between {y[0]} meters and {y[1]} meters"
        else:
            dire = "stay at same"
            new_area_range = None

        movement_block = f"""
    Previous scene objects: {prev_objects_summary}
    Movement Note: You {dire} to absolute ({position}).
    """
        consistency_block = """
    Consistency Note:
    - Objects mentioned in instruction should appear.
    """
    else:
        x = [position[0] - dim_x // 2, position[0] + dim_x // 2]
        y = [position[1], position[1] + dim_y]
        new_area_range = f"x ranging between {x[0]} meters and {x[1]} meters, y ranging between {y[0]} meters and {y[1]} meters"
        movement_block = f"Movement Note: You are at {position}."
        consistency_block = f"""
    Consistency Note:
    - Initialize scene with new objects ONLY in the visible area: {new_area_range}.
    - Generate a scene description.
    """

    planning_prompt = role_block + env_block + placement_order + object_check

    if scene_desc:
        scene_desc_block = f"""
        New Scene Description:
        {scene_desc}
        """
        planning_prompt = planning_prompt + scene_desc_block

    semantic_block = """
    Object Arrangement and Spatial Relationships:
    - Objects should be arranged naturally as seen in real world environment (e.g., chairs around table, cars parked along street, trees spaced in a park, lamp on top of table/desk, pillow on sofa/bed , television facing sofa , vase centered on table ,  bird feeders hanging on a tree, trash bins beside bench ).
    - Define clear spatial relationships between objects (e.g., next to, under, near, on top of, centered on, beside).
    - Align most larger objects along area boundaries(e.g. walls, fences or paths) and corners to reflect natural real-world arrangements and maximise navigable spaces.
    - Ensure bounding boxes and positions are consistent with the defined spatial relationships.

    Follow semantic asset grouping:
    - A semantic asset group is a collection of assets that are logically, functionally, or spatially related within a scene. Assets in a group support a common purpose or activity and together form a meaningful subscene.
    - Group assets based on function (used together), semantics (conceptually related), or geometry (close in space or aligned).
    - Examples:
        - Sleeping area: Bed + Nightstand + Lamp (used together for resting and reading)
        - Entertainment area: Sofa + TV Console + TV  (functionally related for watching)
        - Work area: Desk + Chair + Laptop (supports the working activity)
        - Dining area: Dining Table + Chairs + Chandelier (forms a cohesive eating zone)
    """

    task_block = """
Task:
Decompose this instruction into ATOMIC TASKS for the design process. The output should ONLY be a valid json."""
    planning_prompt = planning_prompt + semantic_block + task_block

    file_path = os.path.join(save_dir, f"planning_prompt_{position[0]}_{position[1]}.txt")
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(planning_prompt)

    return planning_prompt, new_area_range, consistency_block, movement_block


def build_object_placement_prompt(plan_text, position, dim_x, dim_y, consistency_block,
                                  movement_block, env_note, new_area_range):
    role_block = "Role: You are an expert real world planner.\n"
    env_block = f"Environment note: {env_note}.\n"
    return role_block + env_block + f"""

Plan summary for the room:
{plan_text}

Placement Rules:
- You are populating a {dim_x} meters * {dim_y} meters room.
- The room dimensions are x ranging between -{dim_x/2} to {dim_x/2} and y ranging between 0 to {dim_y}. Bounding boxes SHOULD NOT go out of the room area.
- Design such that two BULK objects bounding boxes SHOULD NOT INTERSECT each other unless the relationship is under or on top of.
- While placing, verify that the front of the object being placed and already placed objects remains clearly visible and is not obstructed by any larger object positioned in front of it.
- Maintain realistic distances between the objects you are placing to make the room visually appealing. Do not clutter them together.
- Align most larger objects along area boundaries(e.g. walls, fences or paths) to reflect natural real-world arrangements and maximise navigable spaces.
- Each object must include: label(same as the name mentioned in plan, do not add any adjective), object_type, anchor, description, bbox [x,y,w,h] where (x+w/2,y+h/2) is the position/center, (x,y) is the bottom left corner and (w,h) is the width and height, description, position [x+w/2,y+h/2], rotation angle.
- Object sizes and bounding boxes must be semantically consistent with other objects in the scene and the room dimension.
- Positions and bounding boxes must accurately reflect spatial relationships (e.g., next to, under, on top of, centered on, near).
- Anchored objects must be placed and oriented relative to their reference object according to the defined spatial relationship.

Instructions:
- Output ONLY valid JSON with fields: objects.
"""


def build_orientation_prompt(scene: Dict[str, Any]) -> str:
    objs = scene.get("objects", [])
    obj_lines = "\n".join([
        f"- {o['label']}: {o.get('description', '')} (anchor: {o.get('anchor', 'none')})"
        for o in objs
    ])
    return f"""
You are refining object orientations in a 3D indoor or outdoor scene.
Assign realistic rotation angles (in degrees, 0-360) to each object.

Interpretation:
- 0 deg: facing forward (toward the ego or camera)
- 90 deg: facing right
- 180 deg: facing backward
- 270 deg: facing left

Guidelines:
- Objects attached to walls should face away from the wall.
- Sofas and chairs should face TVs or tables if present.
- Tables should align with nearby seating.
- TVs and monitors should face the main sitting area.
- Beds should face away from their headboard wall.
- Cabinets, wardrobes, and shelves should be against walls and face outward.
- Appliances (stoves, sinks, fridges) should face the open workspace or usable area.
- Floor lamps, indoor plants, and static decor items usually have 0 deg rotation.
- Benches should face open spaces, walkways, or scenic views.
- Vehicles should face their direction of travel or toward open space.
- Trees, rocks, and natural outdoor elements should maintain 0 deg rotation.
- Freestanding objects in open areas should face the scene center or main viewpoint.
- Groups of similar objects (e.g., multiple chairs) should be aligned symmetrically.
- Desks should face open space or toward the user's seating direction.
- Bathroom fixtures (sinks, toilets) should face the center or entry of the room, not walls.

Scene objects:
{scene}

- Output ONLY valid JSON with fields: objects exactly similar to input. Do not output reasoning.
""".strip()


def build_scale_prompt(scene: Dict[str, Any]) -> str:
    print(scene, "Scene build")
    objs = scene.get("objects", [])
    obj_labels = ", ".join([o["label"] for o in objs])
    return f"""
You are refining object scales in a realistic 3D scene.

Adjust the size of each object's bounding box based on its natural proportion relative to surrounding objects in the scene. Maintain realistic scale relationships as seen in real environments.
Ensure relative size consistency - smaller items (e.g., lamps, books, vases) should not exceed larger reference objects (e.g., tables, beds, sofas).
Examples:
- A chair should be smaller than a desk/table.
- A desk should be smaller than a bed.
- A nightstand should be lower than a bed.
- A sofa should be larger than a side table.
- Trees should be taller than nearby benches or cars.
- Buildings should dwarf trees and vehicles.
Keep proportions coherent within the environment (indoor or outdoor), adjusting bounding box dimensions for necessary objects so all objects appear physically plausible when viewed together.

Scene objects:
{scene}

- Output ONLY valid JSON with fields: objects exactly similar to input.
""".strip()


# =============================================================================
# Modules  (plain classes -- no dspy.Module, no ChainOfThought)
# =============================================================================
class SceneDescription:
    """Replaces SceneDescriptionGenerator signature + SceneDescription module."""

    def __call__(self, env_note: str, scene_desc: str):
        print("\n Generating new scene description for position ")
        seed = utils_copy_img.generate_random_seed()
        prompt = f"""Role: You generate a concise updated description of the current scene using prior context, environment notes, and predicted area type.

Random seed: {seed}
Previous scene description: {scene_desc}
Environment note: {env_note}

{SCENE_DESC_SCHEMA}"""
        raw = llm_complete(prompt)
        try:
            data = parse_llm_json(raw)
        except ValueError:
            data = {}
        new_scene_desc = data.get("new_scene_description", "No scene generated.")
        new_scene_name = data.get("concise_scene_name", "No scene generated.")
        new_scene_area = data.get("scene_area", "No scene generated.")
        return new_scene_desc, new_scene_name, new_scene_area


class ObjectChoice:
    """Replaces ObjectChoicePlanner signature + ObjectChoice module."""

    def suggest_objects(self, env_note: str, scene_description: str) -> Dict[str, List[str]]:
        seed = utils_copy_img.generate_random_seed()
        choice_restriction_prompt = """
            Strictly follow these rules:
- Objects not to include:
    - windows, doors, walls, partitions or anything attached to it.
    - floors, roofs.
    - roads, street, pavement, footpath, sidewalk, pathway,lawn and any other part of ground.
    - objects like counter that has multiple english meanings.
- Include only specific objects, not generalized names like new_object, appliances etc.
- Do not include any objects in plural.
- Do not repeat any object more than twice. Try to avoid repeating objects.
        """
        prompt = f"""Role: Decide potential bulk and ornament objects for the described scene.

Random seed: {seed}
Environment note: {env_note}
Scene description: {scene_description}
Choice restrictions: {choice_restriction_prompt}

bulk_objects: large, structural, or fixed objects (e.g., building, tree, table, sofa, road).
ornament_objects: smaller, decorative, or movable objects (e.g., lamp, plant, trash bin, signboard).

{OBJECT_CHOICE_SCHEMA}"""
        raw = llm_complete(prompt)
        try:
            data = parse_llm_json(raw)
        except ValueError:
            data = {}
        bulk = data.get("bulk_objects") or []
        ornaments = data.get("ornament_objects") or []
        return {"bulk_objects": bulk, "ornament_objects": ornaments}


class PlanAtomicTasks:
    """Replaces AtomicTaskPlanning signature + PlanAtomicTasks module."""

    def __init__(self, save_dir):
        self.save_dir = save_dir

    def forward(self, position, env_note, scene_desc, object_store, prev_scenegraphs,
                dim_x, dim_y, dx, dy):
        planning_prompt, new_area_range, consistency_note, movement_note = build_planning_prompt(
            position=position, env_note=env_note, prev_scenegraphs=prev_scenegraphs,
            save_dir=self.save_dir, dim_x=dim_x, dim_y=dim_y, dx=dx, dy=dy,
            object_store=object_store, scene_desc=scene_desc, ornaments_included=False,
        )

        allowed_objs = utils_copy_img._parse_allowed_objects(planning_prompt)
        print("Inside planner")

        seed = utils_copy_img.generate_random_seed()
        full_prompt = f"""Random seed: {seed}

{planning_prompt}

{ATOMIC_TASK_SCHEMA}"""
        raw = llm_complete(full_prompt)
        print("After planning computation")

        parsed = parse_llm_json(raw)
        # Normalize to {"atomic_tasks": [...]}
        if isinstance(parsed, list):
            parsed_pred = {"atomic_tasks": parsed}
        elif isinstance(parsed, dict) and "atomic_tasks" in parsed:
            parsed_pred = {"atomic_tasks": parsed["atomic_tasks"]}
        else:
            parsed_pred = {"atomic_tasks": coerce_obj_list(parsed)}

        with open(os.path.join(self.save_dir, f"planning_{position[0]}_{position[1]}.json"), "w") as f:
            json.dump(parsed_pred, f, indent=2)
        return parsed_pred, new_area_range, consistency_note, movement_note, allowed_objs


class ObjectPlacementManager:
    """
    Replaces ObjectPlacementGenerator signature + dspy.Refine.
    Manual best-of-N refinement loop with feedback re-prompting.
    """

    def __init__(self, save_dir, N=5, threshold=1.0):
        self.save_dir = save_dir
        self.N = N
        self.threshold = threshold
        self._refine_logs = []

    def _generate_placement(self, instruction_prompt, feedback, seed):
        parts = [f"Random seed: {seed}", instruction_prompt]
        if feedback:
            parts.append("Feedback based on previous placements. Follow this feedback:\n" + feedback)
        parts.append(SCENE_OBJECT_SCHEMA)
        prompt = "\n\n".join(parts)
        raw = llm_complete(prompt)
        parsed = parse_llm_json(raw)
        return coerce_obj_list(parsed)

    def forward(self, position, plan, new_area_range, allowed_objs, dim_x, dim_y,
                consistency_block, movement_block, base_scene, dx, dy, env_note="No note"):
        plan_text = "\n".join(
            task.get('execution_prompt', '')
            for task in plan.get("atomic_tasks", [])
            if task.get('execution_prompt')
        )
        instruction_prompt = build_object_placement_prompt(
            plan_text, position, dim_x, dim_y, consistency_block, movement_block,
            env_note, new_area_range,
        )

        new_area_range_parsed = utils_copy_img._parse_new_area_range(new_area_range)

        best_score = float("-inf")
        best_delta: List[Dict[str, Any]] = []
        feedback = ""
        self._refine_logs = []

        for i in range(self.N):
            seed = utils_copy_img.generate_random_seed()
            delta = self._generate_placement(instruction_prompt, feedback, seed)

            # --- visualize this candidate ---
            tmp_scene = copy.deepcopy(base_scene)
            delta_norm = [utils_copy_img.normalize_obj(o) for o in delta]
            tmp_scene["objects"].extend(delta_norm)
            utils_copy_img.visualize(
                tmp_scene, position=position, save_dir=self.save_dir,
                save_name=f"check_{i + 1}", sem_name=f"sem_check_{i + 1}"
            )

            # --- score candidate ---
            score, diagnostics = utils_copy_img.score_candidate_delta(
                delta, base_scene, allowed_objs, new_area_range_parsed, position, dx, dy,
            )

            feedback = (
                "Placement Diagnostics / Issues:\n"
                + "\n".join(f"- {d}" for d in diagnostics)
                + "\n\nAction Required:\n"
                "Update the positions or bounding box dimensions of the objects having "
                "placement problems to address the above issues. Ensure that any changes "
                "do not cause bounding box overlaps with other objects in the scene."
            )

            self._refine_logs.append({
                "iteration": i + 1,
                "delta": delta_norm,
                "score": score,
                "diagnostics": diagnostics,
                "feedback_text": feedback,
            })
            log_path = os.path.join(self.save_dir, f"refine_logs__{position[0]}_{position[1]}.json")
            with open(log_path, "w") as f:
                json.dump(self._refine_logs, f, indent=4)

            if score > best_score:
                best_score = score
                best_delta = delta

            if score >= self.threshold:
                break

        # --- commit the best candidate ---
        delta_objects = [utils_copy_img.normalize_obj(o) for o in best_delta]
        if delta_objects:
            delta_objects = utils_copy_img.assign_progressive_ids(
                delta_objects, scene_graph_dir=self.save_dir,
            )
        base_scene["objects"].extend(delta_objects)
        return base_scene


class OrientationRefinerModule:
    """Replaces OrientationRefiner signature + OrientationRefinerModule (multimodal)."""

    def __init__(self, save_dir):
        self.save_dir = save_dir

    def forward(self, scene: Dict[str, Any], instruction) -> List[Dict[str, Any]]:
        print("\n[OrientationRefiner] Refining object rotations...")
        updated_scene = copy.deepcopy(scene)
        objects = updated_scene.get("objects", [])
        if not objects:
            print("No objects found for orientation refinement.")
            return objects

        for obj in scene.get("objects", []):
            if "rotation_angle" not in obj or obj["rotation_angle"] is None:
                obj["rotation_angle"] = 0.0

        prompt = build_orientation_prompt({"objects": objects})

        json_path = os.path.join(self.save_dir, "check_tmp.json")
        with open(json_path, "w") as f:
            json.dump(scene, f, indent=2)
        print(f"[Saved delta JSON] {json_path}")

        blender_dir = os.path.join(self.save_dir, "check_tmp")
        # Requires `import render_blender` at top to be uncommented.
        render_blender.run_blender(
            json_path=json_path,
            glb_directory="asset_library",
            output_directory=blender_dir,
            environment_name=instruction,
        )

        images = [
            pil_to_b64(Image.open(os.path.join(blender_dir, "view_00.png"))),  # side 1
            pil_to_b64(Image.open(os.path.join(blender_dir, "view_02.png"))),  # side 2
            pil_to_b64(Image.open(os.path.join(blender_dir, "view_01.png"))),  # front
            pil_to_b64(Image.open(os.path.join(blender_dir, "view_03.png"))),  # back
            pil_to_b64(Image.open(os.path.join(blender_dir, "view_04.png"))),  # top
        ]

        seed = utils_copy_img.generate_random_seed()
        full_prompt = f"""Random seed: {seed}

{prompt}

Current placement JSON:
{json.dumps(objects, indent=2)}

The five attached images are: side view 1, side view 2, front view, back view, top view.
Refer to them to critique and correct each object's rotation_angle.
Return the SAME list of objects with ONLY rotation_angle updated.

{SCENE_OBJECT_SCHEMA}"""

        raw = llm_complete(full_prompt, images=images)
        parsed = parse_llm_json(raw)
        return coerce_obj_list(parsed)


class ScaleRefinerModule:
    """Replaces ScaleRefiner signature + ScaleRefinerModule."""

    def __init__(self, save_dir):
        self.save_dir = save_dir

    def forward(self, scene: Dict[str, Any]) -> List[Dict[str, Any]]:
        print("\n[ScaleRefiner] Refining object scales...")
        updated_scene = copy.deepcopy(scene)
        objects = updated_scene.get("objects", [])
        if not objects:
            print("No objects found for scale refinement.")
            return objects

        prompt = build_scale_prompt({"objects": objects})
        seed = utils_copy_img.generate_random_seed()
        full_prompt = f"""Random seed: {seed}

{prompt}

Current placement JSON:
{json.dumps(objects, indent=2)}

Return the SAME list of objects with ONLY bbox and position updated.

{SCENE_OBJECT_SCHEMA}"""

        raw = llm_complete(full_prompt)
        parsed = parse_llm_json(raw)
        return coerce_obj_list(parsed)


# =============================================================================
# Orchestrator
# =============================================================================
class ScenePlanner:
    def __init__(self, save_dir):
        self.scene_desc = SceneDescription()
        self.manager = ObjectPlacementManager(save_dir=save_dir)
        self.object_choice = ObjectChoice()
        self.planner = PlanAtomicTasks(save_dir=save_dir)
        self.orientation_refiner = OrientationRefinerModule(save_dir)
        self.scale_refiner = ScaleRefinerModule(save_dir)
        self.save_dir = save_dir

    def forward(self, position, prev_scenegraphs, instruction, objects_list, dim_x, dim_y,
                dx, dy, orientation=False, scale=False):

        env_note = f"Instruction: {instruction}" if instruction else "No special instruction."

        scene_desc, scene_name, scene_area = self.scene_desc(
            env_note=env_note,
            scene_desc=prev_scenegraphs[-1].get("scene_description", "")
            if len(prev_scenegraphs) != 0 else "No previous scene description!",
        )

        objects = self.object_choice.suggest_objects(env_note, scene_desc)

        plan, new_area_range, consistency_note, movement_note, allowed_objs = self.planner.forward(
            position=position, env_note=env_note, scene_desc=scene_desc,
            object_store=objects_list, prev_scenegraphs=prev_scenegraphs,
            dim_x=dim_x, dim_y=dim_y, dx=dx, dy=dy,
        )

        scene = self.manager.forward(
            position, plan, new_area_range, allowed_objs, dim_x, dim_y,
            consistency_note, movement_note, base_scene={"objects": []},
            dx=dx, dy=dy, env_note=env_note,
        )
        utils_copy_img.visualize(scene, position=position, save_dir=self.save_dir, save_name="initial", sem_name="sem_initial")

        # Refinement passes (objects are plain dicts now)
        if orientation:
            scene_tmp = self.orientation_refiner.forward(scene, instruction)
            scene["objects"] = [o for o in scene_tmp if o.get("label") != "ego_person"]
            utils_copy_img.visualize(scene, position=position, save_dir=self.save_dir, save_name="orient")
        if scale:
            scene_tmp = self.scale_refiner.forward(scene)
            scene["objects"] = [o for o in scene_tmp if o.get("label") != "ego_person"]
            utils_copy_img.visualize(scene, position=position, save_dir=self.save_dir, save_name="scale")

        scene["Environmental_note"] = env_note
        scene["scene_description"] = scene_desc
        scene["Environment Name"] = scene_name
        scene["Scene Area"] = scene_area

        print(scene, "Scene")

        ego_obj = {
            "label": "ego_person",
            "object_type": "bulk",
            "anchor": None,
            "label_index": "ego_person",
            "position": [position[0], position[1]],
            "bbox": [position[0], position[1], 0, 0],
            "description": "Ego agent position",
            "rotation_angle": 0.0,
        }
        scene["objects"].append(ego_obj)

        with open(os.path.join(self.save_dir, f"scene_pos_{position[0]}_{position[1]}.json"), "w") as f:
            json.dump(scene, f, indent=2)

        return scene


# =============================================================================
# Main
# =============================================================================
if __name__ == "__main__":

    save_dir = input("Enter the path to save/load scenegraphs: ").strip()
    os.makedirs(save_dir, exist_ok=True)

    scene_planner = ScenePlanner(save_dir=save_dir)

    all_prev_scenes = []
    pos = (0, 0)

    prev_scene = {
        "objects": [],
        "scene_description": "No previous scene description.",
    }

    while True:
        instr = input("\nEnter environment instruction (or 'quit'): ").strip()
        if instr.lower() == "quit":
            break

        dx, dy = utils_copy_img.extract_movement_from_note(instr)
        abs_x = pos[0] + dx * utils_copy_img.STEP_SIZE
        abs_y = pos[1] + dy * utils_copy_img.STEP_SIZE
        pos = (abs_x, abs_y)
        print("Ego moved to:", pos)

        scenegraph = utils_copy_img.load_existing_scenegraph(pos, save_dir=save_dir)

        if scenegraph:
            print(f"Loaded existing scenegraph for position {pos}")
            result_scene = scenegraph
        else:
            obj_input = input("Enter a list of objects python collect_probe_data.py \
  --benchmark-glob 'benchmarks/*/*.json' \
  --gen-script /path/to/Heuristic_wo_dspy.py \
  --gen-python /path/to/conda/envs/sage/bin/python \
  --gen-save-root gen_runs --num-generations 3 \
  --out-dir probe_data --model-path Qwen/Qwen3.5-27B \
  --grid-rows 5 --grid-cols 5 --probe-target incremental(comma-separated, e.g., 'chair, table, lamp'): ").strip()
            dim_x = int(input("x dimension of the room.").strip())
            dim_y = int(input("y dimension of the room.").strip())
            result_scene = scene_planner.forward(
                position=pos,
                prev_scenegraphs=all_prev_scenes,
                instruction=instr,
                objects_list=obj_input,
                dim_x=dim_x,
                dim_y=dim_y,
                dx=dx,
                dy=dy,
                orientation=False, scale=False,
            )

        utils_copy_img.plot_scenegraph(result_scene, pos, save_dir=save_dir)
        print(f"Scene at position {pos} saved and plotted.")
        utils_copy_img.visualize(result_scene, position=pos, save_dir=save_dir, save_name="final", sem_name="sem_final")
        prev_scene = result_scene
        all_prev_scenes.append(result_scene)