#!/usr/bin/env python3
"""
Extract initial stroke points using SLDvec.

Usage:
    conda activate nsplat
    python extract.py --input data/1.png --output data/1.npz
"""

import argparse
import pathlib
import os
from pathlib import Path
from typing import List, Optional

import numpy as np

# SLDvec imports
from SLDvec.preprocessing import binarize_image, load_image, potrace_vectorize
from SLDvec.ordering import get_predictor, ModelPredictor, get_stroke_order
from SLDvec.skeleton import get_medial_axis
from SLDvec.utils.networkx import get_path_from_degree_1_node_to_crossroad
import networkx as nx


# Adapted from SLDvec, removing splitting logic
def filter_points(G, terminating_node: List[int], node_list: List[int]):
    # Remove the points that are too close to a crossing node
    # Also remove the nodes that are between a non ending single neighbor node and a crossing node
    all_node_pos = np.array([G.nodes[n]["pos"] for n in G.nodes()])
    all_node = list(G.nodes())

    # Get the nodes to remove and the ones where we need to split the curve at
    to_remove = set()
    split_at = []
    for node in node_list:
        if G.degree[node] > 2:
            # Remove points that are too close to a neighbor of a crossroad node
            for n in G.neighbors(node):
                dist = G.nodes[n]["dist"] if "dist" in G.nodes[n] else 0

                # First find all the nodes that are close enough, that is the node inside the square
                infinite_norm = np.max(
                    np.abs(all_node_pos - G.nodes[n]["pos"][None, :]), axis=1
                )
                close_enough = np.where(infinite_norm < dist)[0]

                # Then find the nodes that are inside the circle
                l2_norm = np.linalg.norm(
                    all_node_pos[close_enough] - G.nodes[n]["pos"][None, :], axis=1
                )
                close = close_enough[l2_norm < dist]

                to_remove.update([all_node[idx] for idx in close])

        if G.degree[node] == 1 and node not in terminating_node:
            # Remove points that are between a non ending single neighbor node and a crossing node
            # Also split the path at the single neighbor node
            branch_to_remove = get_path_from_degree_1_node_to_crossroad(G, node)[1:]
            to_remove.update(branch_to_remove)

            split_at.append(node)

    # Keep the nodes at intersection only if the intersection is crossing
    for node in node_list:
        if G.degree[node] == 4 and G.nodes[node]["intersection_type"] == "tangent":
            to_remove.add(node)
    # Keep the node where we need to split the curve at
    for node in split_at:
        if node in to_remove:
            to_remove.discard(node)

    # Remove the nodes from the list
    if len(to_remove) == 0:
        node_list_filtered = node_list
    else:
        node_list_filtered = [n for n in node_list if n not in to_remove]

    # Split the path at the degree 1 node
    # This part is removed

    return node_list_filtered


# Adapted from SLDvec
def get_order(
    image_path: Path,
    intersection_predictor,
    thresh: Optional[float] = None,
    multiple_lines: bool = False,
):
    try:
        # Load the image
        image, orig_image_shape, scale_ratio = load_image(image_path)  # type: ignore
        # Preprocess the image
        binary_image, threshold = binarize_image(image, thresh=thresh)

        # Get the medial axis
        print("SLDVec: Computing the medial axis...")
        curves = potrace_vectorize(binary_image)
        _, simplified_medial_axis = get_medial_axis(
            curves, multiple_lines=multiple_lines
        )

        # Order the graph
        print("SLDVec: Ordering the graph...")
        if not multiple_lines:
            simplified_medial_axis = simplified_medial_axis.subgraph(
                max(nx.connected_components(simplified_medial_axis), key=len)
            )
        node_lists, terminating_node = get_stroke_order(
            G=simplified_medial_axis,
            image=image,
            model=intersection_predictor,
            force_single_line=not multiple_lines,
        )

        return node_lists, terminating_node, simplified_medial_axis

    except Exception as e:
        print("An error occurred during SLDVec processing")
        raise


def get_args():
    parser = argparse.ArgumentParser(description="Extract stroke points using SLDvec")
    parser.add_argument(
        "--dir",
        type=str,
        default=None,
        help="Directory containing input images. Processes all images and saves .npz alongside each.",
    )
    parser.add_argument(
        "--input",
        type=str,
        default="data/1.png",
        help="Input image path (used when --dir is not specified)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output .npz file path (default: same as input with .npz extension)",
    )
    parser.add_argument(
        "--sample-rate",
        type=float,
        default=0.01,
        help="Maximum number of points per stroke after downsampling",
    )
    parser.add_argument(
        "--minimum-points",
        type=int,
        default=30,
        help="Minimum number of points per stroke",
    )
    parser.add_argument(
        "--thresh",
        type=float,
        default=None,
        help="Threshold for binarization (None for auto)",
    )
    parser.add_argument(
        "--single",
        action="store_true",
        help="Assume single-line drawing",
    )
    args = parser.parse_args()
    return args


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif", ".webp"}


def run_once(
    input_file: pathlib.Path,
    output_file: pathlib.Path,
    intersection_predictor,
    sample_rate: float = 0.01,
    minimum_points: int = 10,
    thresh: Optional[float] = None,
    multiple_lines: bool = False,
):
    """Process a single image: extract strokes and save to .npz."""
    print(f"Running SLDvec on {input_file}...")
    node_lists, terminating_node, graph = get_order(
        input_file,
        intersection_predictor=intersection_predictor,  # type: ignore
        thresh=thresh,
        multiple_lines=multiple_lines,
    )

    node_lists_filtered = [
        filter_points(graph, terminating_node, node_list)  # type: ignore
        for node_list in node_lists
    ]
    node_lists_filtered = [x for x in node_lists_filtered if x != [[]] and x != []]

    pts_list = []
    for node_list in node_lists_filtered:
        pts = np.array([graph.nodes[node]["pos"] for node in node_list])
        # Downsample to initial_sample_size
        sample_size = max(minimum_points, int(len(pts) * sample_rate))
        if len(pts) > sample_size:
            indices = np.linspace(0, len(pts) - 1, sample_size, dtype=int)
            pts = pts[indices]
        pts_list.append(pts)

    print(f"Found {len(pts_list)} stroke(s)")
    for i, pts in enumerate(pts_list):
        print(f"  Stroke {i}: {len(pts)} points")

    # Save to npz file
    save_dict = {f"stroke_{i}": pts for i, pts in enumerate(pts_list)}
    save_dict["num_strokes"] = np.array([len(pts_list)])
    output_file.parent.mkdir(parents=True, exist_ok=True)
    np.savez(output_file, **save_dict)
    print(f"Saved points to {output_file}")


def run_dir(
    input_dir: pathlib.Path,
    intersection_predictor,
    sample_rate: float = 0.01,
    minimum_points: int = 10,
    thresh: Optional[float] = None,
    multiple_lines: bool = False,
):
    """Process all images in a directory, saving .npz files alongside inputs."""
    image_files = sorted(
        f
        for f in input_dir.iterdir()
        if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not image_files:
        print(f"No image files found in {input_dir}")
        return

    print(f"Found {len(image_files)} image(s) in {input_dir}")
    for i, img_path in enumerate(image_files):
        output_file = img_path.with_suffix(".npz")
        if output_file.exists():
            print(
                f"[{i+1}/{len(image_files)}] Skipping {img_path.name} (output exists)"
            )
            continue
        print(f"\n[{i+1}/{len(image_files)}] Processing {img_path.name}")
        try:
            run_once(
                input_file=img_path,
                output_file=output_file,
                intersection_predictor=intersection_predictor,
                sample_rate=sample_rate,
                minimum_points=minimum_points,
                thresh=thresh,
                multiple_lines=multiple_lines,
            )
        except Exception as e:
            print(f"  ERROR processing {img_path.name}: {e}")

    print(f"\nDone. Processed {len(image_files)} image(s) from {input_dir}.")


def main():
    args = get_args()
    intersection_predictor = get_predictor()

    multiple_lines = not args.single
    if args.dir is not None:
        run_dir(
            input_dir=pathlib.Path(args.dir),
            intersection_predictor=intersection_predictor,
            sample_rate=args.sample_rate,
            minimum_points=args.minimum_points,
            thresh=args.thresh,
            multiple_lines=multiple_lines,
        )
    else:
        input_file = pathlib.Path(args.input)
        if args.output is None:
            output_file = input_file.with_suffix(".npz")
        else:
            output_file = pathlib.Path(args.output)
        run_once(
            input_file=input_file,
            output_file=output_file,
            intersection_predictor=intersection_predictor,
            sample_rate=args.sample_rate,
            minimum_points=args.minimum_points,
            thresh=args.thresh,
            multiple_lines=multiple_lines,
        )


if __name__ == "__main__":
    main()
