import subprocess
import os
import json
import base64

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np


import argparse
import os
from typing import List, Tuple

import cv2


def plot_bboxes(objs, position, annotate_coords=False, show_centers=False, figsize=(6, 6), linewidth=1.5, env_name="", save_dir=None, save_name=None):
    fig, ax = plt.subplots(figsize=figsize)
    xs, ys = [], []
    labels_to_place = []
    
    for i, o in enumerate(objs):
        x1, y1, x2, y2 = o["x1"], o["y1"], o["x2"], o["y2"]
        xmin, xmax = sorted([x1, x2])
        ymin, ymax = sorted([y1, y2])
        w, h = xmax - xmin, ymax - ymin
        
        rect = Rectangle((xmin, ymin), w, h, fill=False, linewidth=linewidth)
        ax.add_patch(rect)
        xs += [xmin, xmax]; ys += [ymin, ymax]
        
        cx, cy = (xmin + xmax) / 2.0, (ymin + ymax) / 2.0
        
        label = f'{o["text"]}'
        if annotate_coords:
            label += f' [{xmin:.1f}, {ymin:.1f}, {xmax:.1f}, {ymax:.1f}]'
        
        labels_to_place.append({
            'label': label,
            'center': (cx, cy),
            'bbox': (xmin, ymin, xmax, ymax),
            'width': w,
            'height': h
        })
        
        if show_centers:
            ax.scatter([cx], [cy], s=20, zorder=5, color='red', alpha=0.7)
    
    labels_to_place.sort(key=lambda x: -x['center'][1])
    
    placed_labels = []
    
    for label_info in labels_to_place:
        cx, cy = label_info['center']
        label = label_info['label']
        xmin, ymin, xmax, ymax = label_info['bbox']
        w, h = label_info['width'], label_info['height']
        
        positions = [
            (cx, cy, 0, 0),
            (cx, cy + h*0.2, 0, 10),
            (cx, cy - h*0.2, 0, -10),
            (cx - w*0.2, cy, -10, 0),
            (cx + w*0.2, cy, 10, 0),
            (xmin + w*0.2, cy, 0, 0),
            (xmax - w*0.2, cy, 0, 0),
            (cx, ymin + h*0.2, 0, 0),
            (cx, ymax - h*0.2, 0, 0),
        ]
        
        best_pos = positions[0]
        min_overlap = float('inf')
        
        for pos_x, pos_y, offset_x, offset_y in positions:
            if xmin <= pos_x <= xmax and ymin <= pos_y <= ymax:
                overlap_score = 0
                for placed_x, placed_y, placed_w, placed_h in placed_labels:
                    label_w = len(label) * 0.06
                    label_h = 0.3
                    
                    dx = abs(pos_x - placed_x)
                    dy = abs(pos_y - placed_y)
                    
                    if dx < (label_w + placed_w) / 2 and dy < (label_h + placed_h) / 2:
                        overlap_score += (1 - dx / ((label_w + placed_w) / 2)) * (1 - dy / ((label_h + placed_h) / 2))
                    else:
                        overlap_score -= 0.1 * (dx + dy)
                
                if overlap_score < min_overlap:
                    min_overlap = overlap_score
                    best_pos = (pos_x, pos_y, offset_x, offset_y)
        
        pos_x, pos_y, offset_x, offset_y = best_pos
        
        ax.annotate(label, (pos_x, pos_y),
                   textcoords="offset points", xytext=(offset_x, offset_y),
                   ha='center', va='center', fontsize=8,
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="white", 
                           edgecolor="gray", alpha=0.8))
        
        label_w = len(label) * 0.06
        label_h = 0.3
        placed_labels.append((pos_x, pos_y, label_w, label_h))
    
    ax.set_xlabel("x"); ax.set_ylabel("y")
    ax.set_title(env_name)
    ax.grid(True, which="both", linestyle="--", linewidth=0.5)
    ax.set_aspect("equal")
    
    if xs and ys:
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)
        
        x_min_lim = min(0, min_x - 0.5)
        x_max_lim = max(10, max_x + 0.5)
        y_min_lim = min(0, min_y - 0.5)
        y_max_lim = max(10, max_y + 0.5)
        
        ax.set_xlim(x_min_lim, x_max_lim)
        ax.set_ylim(y_min_lim, y_max_lim)
    else:
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 10)
    
    ax.minorticks_on()
    ax.grid(True, which='minor', linestyle=':', linewidth=0.3, alpha=0.5)
    
    if save_dir:
        filename = f"{save_name}_{position[0]}_{position[1]}.png"
        fig.savefig(os.path.join(save_dir, filename), bbox_inches='tight', dpi=150)
        print(f"Saved plot to {save_dir}")


def generate_objects_prompt(objects_list):
    object_descriptions = []
    
    for obj in objects_list:
        width = abs(obj['x2'] - obj['x1'])
        height = abs(obj['y2'] - obj['y1'])
        
        object_name = obj['text'].replace('_', ' ')
        
        description = f"object: {object_name}"
        object_descriptions.append(description)
    
    return '\n'.join(object_descriptions)

def create_imagen_request(objects_list):
    objects_text = generate_objects_prompt(objects_list)
    
    prompt = f"""You are an expert assistant who generates 2D icons for a 2D scene.
Your task is to generate a single image that contains cute icons for the list of objects provided below. The output should be a single image.
Instructions:
Layout: Arrange the icons horizontally in a single row, from left to right.
Canvas Size: The overall image canvas size should be calculated to fit all objects and their margins horizontally (one next to each other).
Canvas Background: solid white.
Objects to generate:
{objects_text}. Do not stack the icons, place them horizontally right next to each other. Do not include textual descriptions or numbers in the generated image. Show only the icons, no text, no dimensions. Include ALL, include big objects."""
    
    request_data = {
        "instances": [
            {"prompt": prompt}
        ],
        "parameters": {
            "sampleCount": 1,
            "aspectRatio": "1:1",
            "safetySetting": "block_medium_and_above",
            "personGeneration": "dont_allow",
            "addWatermark": True
        }
    }
    
    return request_data


def find_icon_boxes(
    binary_mask: np.ndarray, dilation_kernel: Tuple[int, int] = (10, 5)
) -> List[Tuple[int, int, int, int]]:
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, dilation_kernel)
    dilated = cv2.dilate(binary_mask, kernel, iterations=1)

    contours, _ = cv2.findContours(
        dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    boxes = [cv2.boundingRect(c) for c in contours]
    boxes.sort(key=lambda b: b[0])
    return boxes


def merge_nearby_boxes(boxes: List[Tuple[int, int, int, int]], factor: float = 0.5) -> List[Tuple[int, int, int, int]]:
    if not boxes:
        return []
    heights = [h for (_, _, _, h) in boxes]
    median_h = np.median(heights)
    gap_threshold = factor * median_h

    merged: List[Tuple[int, int, int, int]] = []
    current = boxes[0]
    for bx in boxes[1:]:
        x, y, w, h = bx
        cx, cy, cw, ch = current
        gap = x - (cx + cw)
        if gap <= gap_threshold:
            nx0 = min(cx, x)
            ny0 = min(cy, y)
            nx1 = max(cx + cw, x + w)
            ny1 = max(cy + ch, y + h)
            current = (nx0, ny0, nx1 - nx0, ny1 - ny0)
        else:
            merged.append(current)
            current = bx
    merged.append(current)
    return merged


def split_icons(
    image_path: str, output_dir: str, margin: int = 5, dilation_kernel: Tuple[int, int] = (20, 5), object_names=[]
) -> List[str]:
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"Unable to read image: {image_path}")

    if img.shape[-1] == 4:
        rgb = img[:, :, :3]
        gray = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
    else:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    _, binary_inv = cv2.threshold(gray, 250, 255, cv2.THRESH_BINARY_INV)

    boxes = find_icon_boxes(binary_inv, dilation_kernel=dilation_kernel)
    boxes = merge_nearby_boxes(boxes, factor=0.1)

    saved_paths = []
    for idx, (x, y, w, h) in enumerate(boxes, start=0):
        x0 = max(0, x - margin)
        y0 = max(0, y - margin)
        x1 = min(img.shape[1], x + w + margin)
        y1 = min(img.shape[0], y + h + margin)
        crop = img[y0:y1, x0:x1]
        out_path = os.path.join(output_dir, f"{object_names[idx]}.png")
        cv2.imwrite(out_path, crop)
        saved_paths.append(out_path)
    return saved_paths


from PIL import Image

def remove_white_background(input_path, output_path, threshold=240):
    img = Image.open(input_path)
    
    img = img.convert("RGBA")
    
    data = img.getdata()
    
    new_data = []
    for item in data:
        if item[0] > threshold and item[1] > threshold and item[2] > threshold:
            new_data.append((255, 255, 255, 0))
        else:
            new_data.append(item)
    
    img.putdata(new_data)
    
    img.save(output_path, "PNG")
    print(f"Saved transparent image to {output_path}")

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
import numpy as np
from PIL import Image
import os

def plot_bboxes_with_icons(objs, position, save_dir, icon_path="path", annotate_coords=True, show_centers=True, 
                           figsize=(6, 6), linewidth=1.5, env_name="", fallback_to_rect=True,
                           padding_factor=0.9, show_bbox_outline=False, show_labels=True):
    fig, ax = plt.subplots(figsize=figsize)
    xs, ys = [], []
    labels_to_place = []
    
    dpi = fig.dpi
    axis_width_inches = figsize[0]
    axis_height_inches = figsize[1]
    
    for i, o in enumerate(objs):
        x1, y1, x2, y2 = o["x1"], o["y1"], o["x2"], o["y2"]
        xmin, xmax = sorted([x1, x2])
        ymin, ymax = sorted([y1, y2])
        w, h = xmax - xmin, ymax - ymin
        xs += [xmin, xmax]; ys += [ymin, ymax]
        
        cx, cy = (xmin + xmax) / 2.0, (ymin + ymax) / 2.0
        
        if show_bbox_outline:
            rect = Rectangle((xmin, ymin), w, h, fill=False, linewidth=linewidth*0.5, 
                           edgecolor='lightgray', linestyle=':', alpha=0.5, zorder=1)
            ax.add_patch(rect)
        
        icon_file = os.path.join(icon_path, f"{o['text']}.png")
        icon_loaded = False
        
        if os.path.exists(icon_file):
            try:
                img = Image.open(icon_file)
                
                if img.mode != 'RGBA':
                    img = img.convert('RGBA')
                
                img_w, img_h = img.size
                
                img_aspect = img_w / img_h
                bbox_aspect = w / h
                
                if img_aspect > bbox_aspect:
                    target_width_pixels = w * dpi * axis_width_inches / 10
                    zoom = (target_width_pixels / img_w) * padding_factor
                else:
                    target_height_pixels = h * dpi * axis_height_inches / 10
                    zoom = (target_height_pixels / img_h) * padding_factor
                
                imagebox = OffsetImage(img, zoom=zoom)
                
                ab = AnnotationBbox(imagebox, (cx, cy), frameon=False, zorder=2)
                ax.add_artist(ab)
                
                icon_loaded = True
                print(f"✓ Loaded '{o['text']}' - Original: {img_w}x{img_h}px, Zoom: {zoom:.3f}")
                
            except Exception as e:
                print(f"✗ Failed to load icon for '{o['text']}': {e}")
        else:
            print(f"✗ Icon not found for '{o['text']}' at {icon_file}")
        
        if not icon_loaded and fallback_to_rect:
            rect = Rectangle((xmin, ymin), w, h, fill=False, linewidth=linewidth, 
                           edgecolor='red', linestyle='--', zorder=2)
            ax.add_patch(rect)
        
        label = f'{o["text"]}'
        if annotate_coords:
            label += f' [{xmin:.1f}, {ymin:.1f}, {xmax:.1f}, {ymax:.1f}]'
        
        labels_to_place.append({
            'label': label,
            'center': (cx, cy),
            'bbox': (xmin, ymin, xmax, ymax),
            'width': w,
            'height': h,
            'icon_loaded': icon_loaded
        })
        
        if show_centers:
            ax.scatter([cx], [cy], s=20, zorder=10, color='red', alpha=0.7)
    
    labels_to_place.sort(key=lambda x: -x['center'][1])
    
    placed_labels = []
    
    for label_info in labels_to_place:
        cx, cy = label_info['center']
        label = label_info['label']
        xmin, ymin, xmax, ymax = label_info['bbox']
        w, h = label_info['width'], label_info['height']
        icon_loaded = label_info['icon_loaded']
        
        positions = [
            (cx, cy, 0, 0),
            (cx, cy + h*0.25, 0, 10),
            (cx, cy - h*0.25, 0, -10),
            (cx - w*0.25, cy, -10, 0),
            (cx + w*0.25, cy, 10, 0),
            (xmin + w*0.3, cy, 0, 0),
            (xmax - w*0.3, cy, 0, 0),
            (cx, ymin + h*0.3, 0, 0),
            (cx, ymax - h*0.3, 0, 0),
        ]
        
        best_pos = positions[0]
        min_overlap = float('inf')
        
        for pos_x, pos_y, offset_x, offset_y in positions:
            if xmin <= pos_x <= xmax and ymin <= pos_y <= ymax:
                overlap_score = 0
                for placed_x, placed_y, placed_w, placed_h in placed_labels:
                    label_w = len(label) * 0.06
                    label_h = 0.3
                    
                    dx = abs(pos_x - placed_x)
                    dy = abs(pos_y - placed_y)
                    
                    if dx < (label_w + placed_w) / 2 and dy < (label_h + placed_h) / 2:
                        overlap_score += (1 - dx / ((label_w + placed_w) / 2)) * (1 - dy / ((label_h + placed_h) / 2))
                    else:
                        overlap_score -= 0.1 * (dx + dy)
                
                if overlap_score < min_overlap:
                    min_overlap = overlap_score
                    best_pos = (pos_x, pos_y, offset_x, offset_y)
        
        pos_x, pos_y, offset_x, offset_y = best_pos
        
        bbox_props = dict(boxstyle="round,pad=0.2", facecolor="white", 
                         edgecolor="gray" if icon_loaded else "red", 
                         alpha=0.8, linewidth=0.5)
        
        if show_labels:
            ax.annotate(label, (pos_x, pos_y),
                    textcoords="offset points", xytext=(offset_x, offset_y),
                    ha='center', va='center', fontsize=7,
                    bbox=bbox_props, zorder=15)
        
        label_w = len(label) * 0.06
        label_h = 0.3
        placed_labels.append((pos_x, pos_y, label_w, label_h))
    
    ax.set_title(env_name)
    ax.grid(True, which="both", linestyle="--", linewidth=0.5, zorder=0)
    ax.set_aspect("equal")
    
    if xs and ys:
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)
        
        x_min_lim = min(0, min_x - 0.5)
        x_max_lim = max(10, max_x + 0.5)
        y_min_lim = min(0, min_y - 0.5)
        y_max_lim = max(10, max_y + 0.5)
        
        ax.set_xlim(x_min_lim, x_max_lim)
        ax.set_ylim(y_min_lim, y_max_lim)
    else:
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 10)
    
    ax.minorticks_on()
    ax.grid(True, which='minor', linestyle=':', linewidth=0.3, alpha=0.5, zorder=0)
    
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    plt.title('')
    plt.savefig(f"{save_dir}/{env_name}_{position[0]}_{position[1]}.png")
    plt.show()

import random

_LABEL_COLOR_CACHE = {}

def _get_color_for_label(label):
    if label not in _LABEL_COLOR_CACHE:
        _LABEL_COLOR_CACHE[label] = (random.random(), random.random(), random.random())
    return _LABEL_COLOR_CACHE[label]

def plot_semplot(objs, position, annotate_coords=False, show_centers=False,
                figsize=(6, 6), linewidth=1.5, env_name="",
                save_dir=None, save_name=None):

    fig, ax = plt.subplots(figsize=figsize)
    xs, ys = [], []

    for o in objs:
        x1, y1, x2, y2 = o["x1"], o["y1"], o["x2"], o["y2"]
        xmin, xmax = sorted([x1, x2])
        ymin, ymax = sorted([y1, y2])
        w, h = xmax - xmin, ymax - ymin

        label_txt = o["text"]
        color = _get_color_for_label(label_txt)

        rect = Rectangle((xmin, ymin), w, h, fill=True,
                         linewidth=linewidth, edgecolor=color,
                         facecolor=color, alpha=0.35)
        ax.add_patch(rect)

        xs += [xmin, xmax]; ys += [ymin, ymax]

        cx, cy = (xmin + xmax)/2.0, ymax

        label = label_txt
        if annotate_coords:
            label += f" [{xmin:.1f},{ymin:.1f},{xmax:.1f},{ymax:.1f}]"

        ax.annotate(label, (cx, ymax),
                    textcoords="offset points", xytext=(0, 5),
                    ha='center', va='bottom', fontsize=8,
                    bbox=dict(boxstyle="round,pad=0.25",
                              facecolor="white", edgecolor=color, alpha=0.9))

        if show_centers:
            ax.scatter([cx], [(ymin+ymax)/2], s=20, zorder=5,
                       color=color, alpha=0.9)

    ax.set_xlabel("x"); ax.set_ylabel("y")
    ax.set_title(env_name)
    ax.grid(True, which="both", linestyle="--", linewidth=0.5)
    ax.set_aspect("equal")

    if xs and ys:
        ax.set_xlim(min(0, min(xs)-0.5), max(10, max(xs)+0.5))
        ax.set_ylim(min(0, min(ys)-0.5), max(10, max(ys)+0.5))
    else:
        ax.set_xlim(0, 10); ax.set_ylim(0, 10)

    ax.minorticks_on()
    ax.grid(True, which='minor', linestyle=':', linewidth=0.3, alpha=0.5)

    if save_dir:
        filename = f"{save_name}_{position[0]}_{position[1]}.png"
        fig.savefig(os.path.join(save_dir, filename), bbox_inches='tight', dpi=150)
        print(f"Saved plot to {save_dir}/{filename}")
