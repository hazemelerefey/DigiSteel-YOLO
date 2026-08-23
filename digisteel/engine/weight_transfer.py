"""
Weight transfer utility for DAFE-integrated models.

When DAFE modules are inserted into a YOLO backbone, all subsequent layer
indices shift, causing Ultralytics' model.load() to fail at key matching.
This utility performs positional weight transfer: it maps base model layers
to DAFE model layers by position, skipping the newly inserted DAFE modules.
"""

import torch
from typing import Dict, List, Tuple


# Mapping: (base_idx, dafe_idx) pairs for YOLOv26n + DAFE
# DAFE inserted at positions 3 and 6 in the DAFE model
YOLOV26N_DAFE_MAPPING = [
    (0, 0),    # Conv (P1)
    (1, 1),    # Conv (P2)
    (2, 2),    # C3k2
    # (3 = DAFE @P2, skip)
    (3, 4),    # Conv (P3)
    (4, 5),    # C3k2
    # (6 = DAFE @P3, skip)
    (5, 7),    # Conv (P4)
    (6, 8),    # C3k2
    (7, 9),    # Conv (P5)
    (8, 10),   # C3k2
    (9, 11),   # SPPF
    (10, 12),  # C2PSA
    (11, 13),  # Upsample
    (12, 14),  # Concat
    (13, 15),  # C3k2
    (14, 16),  # Upsample
    (15, 17),  # Concat
    (16, 18),  # C3k2
    (17, 19),  # Conv
    (18, 20),  # Concat
    (19, 21),  # C3k2
    (20, 22),  # Conv
    (21, 23),  # Concat
    (22, 24),  # C3k2
    (23, 25),  # Detect
]


def transfer_yolov26n_weights_to_dafe(
    base_model: torch.nn.Module,
    dafe_model: torch.nn.Module,
    mapping: List[Tuple[int, int]] = YOLOV26N_DAFE_MAPPING,
    verbose: bool = True,
) -> Dict[str, int]:
    """
    Transfer weights from base YOLOv26n to YOLOv26n+DAFE model.

    Args:
        base_model: The pretrained base YOLOv26n model (source).
        dafe_model: The DAFE-integrated model (destination).
        mapping: List of (base_idx, dafe_idx) pairs.
        verbose: Print transfer details.

    Returns:
        Dict with 'transferred', 'skipped', 'failed' counts.
    """
    base_seq = base_model.model  # Sequential layers
    dafe_seq = dafe_model.model  # Sequential layers

    stats = {"transferred": 0, "skipped": 0, "failed": 0, "dafe_layers": 0}

    # Identify DAFE layer indices in the destination model
    dafe_indices = set()
    for i, layer in enumerate(dafe_seq):
        layer_type = type(layer).__name__
        if "DAFE" in layer_type or "dafe" in layer_type.lower():
            dafe_indices.add(i)
            stats["dafe_layers"] += 1

    if verbose:
        print(f"DAFE layers found at indices: {sorted(dafe_indices)}")
        print(f"Transfer mapping: {len(mapping)} layer pairs")
        print()

    # Validate mapping: check that all target indices have matching source types
    mismatches = []
    for base_idx, dafe_idx in mapping:
        if base_idx < len(base_seq) and dafe_idx < len(dafe_seq):
            bt = type(base_seq[base_idx]).__name__
            dt = type(dafe_seq[dafe_idx]).__name__
            if bt != dt and dafe_idx not in dafe_indices:
                mismatches.append(f"  [{base_idx}->{dafe_idx}]: {bt} -> {dt}")
    if mismatches:
        import warnings
        warnings.warn(
            f"Weight transfer mapping has {len(mismatches)} type mismatches:\n"
            + "\n".join(mismatches),
            stacklevel=2,
        )

    # Transfer weights layer by layer
    for base_idx, dafe_idx in mapping:
        if base_idx >= len(base_seq) or dafe_idx >= len(dafe_seq):
            if verbose:
                print(f"  SKIP [{base_idx}->{dafe_idx}]: index out of range")
            stats["skipped"] += 1
            continue

        base_layer = base_seq[base_idx]
        dafe_layer = dafe_seq[dafe_idx]

        # Verify layer types match
        base_type = type(base_layer).__name__
        dafe_type = type(dafe_layer).__name__

        if base_type != dafe_type:
            if verbose:
                print(f"  SKIP [{base_idx}->{dafe_idx}]: type mismatch ({base_type} vs {dafe_type})")
            stats["skipped"] += 1
            continue

        # Copy state dict
        base_state = base_layer.state_dict()
        dafe_state = dafe_layer.state_dict()

        # Check key compatibility
        common_keys = set(base_state.keys()) & set(dafe_state.keys())
        if not common_keys:
            if verbose:
                print(f"  SKIP [{base_idx}->{dafe_idx}]: no common keys")
            stats["skipped"] += 1
            continue

        # Transfer matching parameters
        transferred_keys = 0
        failed_keys = 0
        for key in common_keys:
            if base_state[key].shape == dafe_state[key].shape:
                dafe_state[key] = base_state[key].clone()
                transferred_keys += 1
            else:
                failed_keys += 1
                if verbose:
                    print(f"  WARN [{base_idx}->{dafe_idx}] {key}: shape mismatch "
                          f"({base_state[key].shape} vs {dafe_state[key].shape})")

        dafe_layer.load_state_dict(dafe_state, strict=False)
        stats["transferred"] += transferred_keys
        stats["failed"] += failed_keys

        if verbose:
            print(f"  OK   [{base_idx:2d}->{dafe_idx:2d}] {base_type:<12} "
                  f"{transferred_keys} params transferred")

    # Summary
    if verbose:
        print()
        print(f"Transfer summary:")
        print(f"  Transferred: {stats['transferred']} parameter tensors")
        print(f"  DAFE layers: {stats['dafe_layers']} (randomly initialized)")
        print(f"  Failed:      {stats['failed']}")
        if stats["transferred"] > 0:
            total_params = sum(p.numel() for p in dafe_model.parameters())
            transferred_params = 0
            for base_idx, dafe_idx in mapping:
                if dafe_idx < len(dafe_seq):
                    for p in dafe_seq[dafe_idx].parameters():
                        transferred_params += p.numel()
            print(f"  Coverage:    ~{transferred_params/total_params*100:.1f}% of parameters")

    return stats


def load_pretrained_to_dafe_model(
    dafe_model_path: str,
    pretrained_weights: str = "yolo26n.pt",
    device: str = "cpu",
) -> torch.nn.Module:
    """
    Convenience function: build DAFE model and load pretrained base weights.

    Args:
        dafe_model_path: Path to the DAFE model YAML config.
        pretrained_weights: Path or name of pretrained base weights.
        device: Device to load on.

    Returns:
        DAFE model with pretrained weights loaded.
    """
    from ultralytics import YOLO

    # Load base model with full pretrained weights
    print(f"Loading base pretrained model: {pretrained_weights}")
    base = YOLO(pretrained_weights)
    base_weights = base.model.state_dict()

    # Build DAFE model from YAML
    print(f"Building DAFE model from: {dafe_model_path}")
    dafe = YOLO(dafe_model_path)

    # Transfer weights
    print("\nTransferring weights...")
    stats = transfer_yolov26n_weights_to_dafe(base.model, dafe.model)

    return dafe
