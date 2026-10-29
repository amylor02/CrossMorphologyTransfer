"""Morphology recognition from BVH skeleton hierarchies.

Only the HIERARCHY block of a BVH file is read: joint names and ordered parent indices.
Offsets, channels, end sites and motion data are ignored, so a signature identifies a
skeleton's topology and naming, not its animation or bone lengths. Recognition is an
exact match against config/skeleton_registry.json. Unknown, malformed or ambiguous
skeletons raise SkeletonError; nothing is guessed.

Command line (from python/):
    python src_ortho_new/skeleton_recognition.py path/to/file.bvh
"""

import argparse
import json
import os
import sys
from dataclasses import dataclass

from architectures import ARCHITECTURE_MODULES

PYTHON_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REGISTRY_PATH = os.path.join(PYTHON_ROOT, "config", "skeleton_registry.json")
REGISTRY_SCHEMA_VERSION = 1


class SkeletonError(RuntimeError):
    """Malformed BVH, unknown or ambiguous skeleton, or an invalid registry."""


@dataclass(frozen=True)
class Skeleton:
    joint_names: tuple
    parents: tuple  # index of each joint's parent; -1 for the root

    @property
    def joint_count(self):
        return len(self.joint_names)


@dataclass(frozen=True)
class RegistryEntry:
    morphology: str
    variant: str
    param_name: str
    architecture: str  # shared by all variants of the morphology
    skeleton: Skeleton


@dataclass(frozen=True)
class Recognition:
    morphology: str
    variant: str
    param_name: str
    architecture: str
    skeleton: Skeleton


def read_skeleton(path):
    """Read the joint names and parent indices from a BVH HIERARCHY block (header only)."""
    names, parents, stack = [], [], []  # stack entries: ("joint", index) or ("site", None)
    saw_hierarchy = saw_motion = False
    try:
        with open(path, "r", encoding="utf-8-sig") as fh:
            for line_no, raw in enumerate(fh, start=1):
                tokens = raw.split()
                if not tokens:
                    continue
                head = tokens[0]
                if not saw_hierarchy:
                    if head != "HIERARCHY":
                        raise SkeletonError(f"{path}: not a BVH file (first token {head!r}, expected HIERARCHY)")
                    saw_hierarchy = True
                    continue
                if head == "MOTION":
                    saw_motion = True
                    break
                if head in ("ROOT", "JOINT"):
                    if len(tokens) < 2:
                        raise SkeletonError(f"{path}: line {line_no}: {head} without a joint name")
                    if head == "ROOT" and names:
                        raise SkeletonError(f"{path}: line {line_no}: ROOT after other joints")
                    if head == "JOINT" and not names:
                        raise SkeletonError(f"{path}: line {line_no}: JOINT before ROOT")
                    parent = next((idx for kind, idx in reversed(stack) if kind == "joint"), -1)
                    names.append(tokens[1])
                    parents.append(parent)
                    stack.append(("joint", len(names) - 1))
                elif head == "End" and len(tokens) > 1 and tokens[1] == "Site":
                    stack.append(("site", None))
                elif head == "}":
                    if not stack:
                        raise SkeletonError(f"{path}: line {line_no}: unbalanced '}}' in hierarchy")
                    stack.pop()
    except UnicodeDecodeError as exc:
        raise SkeletonError(f"{path}: not a text BVH file ({exc.reason})") from exc
    except OSError as exc:
        raise SkeletonError(f"{path}: cannot read file ({exc.strerror})") from exc

    if not saw_hierarchy:
        raise SkeletonError(f"{path}: empty file or missing HIERARCHY section")
    if not names:
        raise SkeletonError(f"{path}: hierarchy has no ROOT joint")
    if stack:
        raise SkeletonError(f"{path}: unclosed block at the end of the hierarchy (truncated file?)")
    if not saw_motion:
        raise SkeletonError(f"{path}: no MOTION section (truncated file?)")
    return Skeleton(tuple(names), tuple(parents))


def _validate_signature(skeleton, where):
    if skeleton.joint_count == 0 or skeleton.parents[0] != -1:
        raise SkeletonError(f"{where}: signature must start with a root joint")
    for index, parent in enumerate(skeleton.parents[1:], start=1):
        if not 0 <= parent < index:
            raise SkeletonError(f"{where}: joint {index} has invalid parent {parent}")


def read_registry(path=REGISTRY_PATH):
    """Read the raw registry JSON. Pass the result to parse_registry() to validate it."""
    if not os.path.isfile(path):
        raise SkeletonError(f"skeleton registry not found: {path}")
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, json.JSONDecodeError) as exc:
        raise SkeletonError(f"skeleton registry {path} is not readable JSON: {exc}") from exc


def load_registry(path=REGISTRY_PATH):
    """Load and validate the registry. Every signature must map to exactly one morphology."""
    return parse_registry(read_registry(path), path)


def parse_registry(data, path=REGISTRY_PATH):
    """Validate registry data (as read by read_registry) and return its entries."""
    if not isinstance(data, dict) or data.get("schema_version") != REGISTRY_SCHEMA_VERSION:
        raise SkeletonError(f"skeleton registry {path}: expected schema_version {REGISTRY_SCHEMA_VERSION}")
    morphologies = data.get("morphologies")
    if not isinstance(morphologies, dict) or not morphologies:
        raise SkeletonError(f"skeleton registry {path}: no morphologies registered")

    entries, owner = [], {}
    for morphology, body in morphologies.items():
        param_name = body.get("param_name") if isinstance(body, dict) else None
        if not isinstance(param_name, str) or not param_name:
            raise SkeletonError(f"skeleton registry: morphology '{morphology}' has no param_name")
        architecture = body.get("architecture") if isinstance(body, dict) else None
        if not isinstance(architecture, str) or architecture not in ARCHITECTURE_MODULES:
            raise SkeletonError(
                f"skeleton registry: morphology '{morphology}' has architecture {architecture!r}; "
                f"supported: {', '.join(sorted(ARCHITECTURE_MODULES))}"
            )
        signatures = body.get("signatures") or []
        if not signatures:
            raise SkeletonError(f"skeleton registry: morphology '{morphology}' has no signatures")
        variants = set()
        for sig in signatures:
            variant = sig.get("variant", "standard")
            where = f"skeleton registry: {morphology}/{variant}"
            if variant in variants:
                raise SkeletonError(f"{where}: duplicate variant name")
            variants.add(variant)
            names = tuple(sig.get("joint_names") or ())
            parents = tuple(sig.get("parents") or ())
            if len(names) != len(parents) or sig.get("joint_count") != len(names):
                raise SkeletonError(f"{where}: joint_count, joint_names and parents disagree")
            skeleton = Skeleton(names, parents)
            _validate_signature(skeleton, where)
            if skeleton in owner:
                other = owner[skeleton]
                raise SkeletonError(f"{where} and {other} share the same signature; recognition would be ambiguous")
            owner[skeleton] = f"{morphology}/{variant}"
            entries.append(RegistryEntry(morphology, variant, param_name, architecture, skeleton))
    return tuple(entries)


def recognize(bvh_path, registry=None):
    """Identify the morphology of a BVH file. Raises SkeletonError if unknown or ambiguous."""
    registry = load_registry() if registry is None else registry
    skeleton = read_skeleton(bvh_path)
    matches = [entry for entry in registry if entry.skeleton == skeleton]
    if len(matches) > 1:
        names = ", ".join(f"{m.morphology}/{m.variant}" for m in matches)
        raise SkeletonError(f"{bvh_path}: skeleton matches several registered entries ({names})")
    if not matches:
        known = sorted({(e.morphology, e.skeleton.joint_count) for e in registry})
        listing = ", ".join(f"{m} ({n} joints)" for m, n in known)
        raise SkeletonError(
            f"{bvh_path}: unknown skeleton ({skeleton.joint_count} joints, root {skeleton.joint_names[0]!r}). "
            f"Registered: {listing}. Register the skeleton in config/skeleton_registry.json or use a supported input."
        )
    entry = matches[0]
    return Recognition(entry.morphology, entry.variant, entry.param_name, entry.architecture, skeleton)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Identify the morphology of BVH files from their hierarchy.")
    parser.add_argument("bvh", nargs="+", help="BVH files to recognize")
    parser.add_argument("--registry", default=REGISTRY_PATH, help="registry JSON (default config/skeleton_registry.json)")
    args = parser.parse_args(argv)
    registry = load_registry(args.registry)
    status = 0
    for path in args.bvh:
        try:
            found = recognize(path, registry)
            print(f"{path}: {found.morphology}/{found.variant} -> {found.param_name} ({found.skeleton.joint_count} joints)")
        except SkeletonError as exc:
            print(f"error: {exc}", file=sys.stderr)
            status = 1
    return status


if __name__ == "__main__":
    sys.exit(main())
