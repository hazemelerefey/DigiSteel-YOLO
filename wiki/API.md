# API Reference

This page documents the key public classes and functions implemented in the current snapshot. All paths below are within the Python package [digisteel/](file:///workspace/digisteel).

## Package Exports (`digisteel`)

The package root re-exports:

- `digisteel.GhostConv` from [digisteel/__init__.py](file:///workspace/digisteel/__init__.py#L31-L37)
- `digisteel.InnerWIoULoss` from [digisteel/__init__.py](file:///workspace/digisteel/__init__.py#L31-L37)

Example:

```python
from digisteel import GhostConv, InnerWIoULoss
```

## A2: GhostConv Modules

Source: [ghost_conv.py](file:///workspace/digisteel/modules/ghost_conv.py)

### `GhostModule`

Core “GhostNet-style” convolution building block.

- **Constructor:** `GhostModule(in_channels, out_channels, kernel_size=1, ratio=2, dw_size=3, stride=1, use_bn=True, act="relu")`
- **Forward:** `forward(x) -> Tensor`
  - `x`: `(N, C_in, H, W)`
  - output: `(N, C_out, H_out, W_out)` where `H_out/W_out` depend on `stride`

Implementation reference: [GhostModule](file:///workspace/digisteel/modules/ghost_conv.py#L19-L85)

### `GhostConv`

Thin wrapper around `GhostModule` intended as a “Conv-like” replacement at call sites.

- **Constructor:** `GhostConv(in_channels, out_channels, kernel_size=3, stride=1, padding=1, groups=1, dilation=1, bias=False)`
- **Forward:** `forward(x) -> Tensor`

Notes:

- `padding`, `groups`, `dilation`, `bias` are accepted but not currently used inside the implementation; the internal block is built via `GhostModule(..., stride=stride)` (see [GhostConv.__init__](file:///workspace/digisteel/modules/ghost_conv.py#L95-L108)).

### `GhostConvWeightSharing`

Holds a single shared `GhostModule` instance to be reused across multiple feature maps (intended for P3/P4/P5 pyramid stages).

- **Constructor:** `GhostConvWeightSharing(in_channels, out_channels, kernel_size=3, ratio=2, dw_size=3)`
- **Forward:** `forward(x) -> Tensor` (delegates to `shared_ghost`)
- **Helper:** `param_count() -> int` counts trainable parameters in the shared module

Implementation reference: [GhostConvWeightSharing](file:///workspace/digisteel/modules/ghost_conv.py#L113-L152)

## A3: Inner-WIoU Loss

Source: [inner_wiou.py](file:///workspace/digisteel/modules/inner_wiou.py)

### Box Format Contract

All box tensors are expected to be shape `(N, 4)` in **XYXY** format:

`[x1, y1, x2, y2]`

### `iou(box1, box2, eps=1e-7) -> Tensor`

Computes IoU for paired rows of `box1` and `box2`.

Implementation reference: [iou](file:///workspace/digisteel/modules/inner_wiou.py#L23-L48)

### `inner_iou_loss(pred_boxes, target_boxes, eps=1e-7) -> Tensor`

Returns a scalar loss (mean over batch) intended to be minimized.

Implementation reference: [inner_iou_loss](file:///workspace/digisteel/modules/inner_wiou.py#L51-L95)

### `wiou_v3_loss(pred_boxes, target_boxes, eps=1e-7) -> Tensor`

Returns a scalar loss (mean over batch) intended to be minimized.

Implementation reference: [wiou_v3_loss](file:///workspace/digisteel/modules/inner_wiou.py#L97-L134)

### `InnerWIoULoss`

Composite loss module:

`L = lambda_weight * inner_iou_loss + (1 - lambda_weight) * wiou_v3_loss`

- **Constructor:** `InnerWIoULoss(lambda_weight=0.5, eps=1e-7)`
- **Forward:** `forward(pred_boxes, target_boxes) -> Tensor` (scalar)
- **Repr:** includes lambda and eps (see [__repr__](file:///workspace/digisteel/modules/inner_wiou.py#L187-L188))

