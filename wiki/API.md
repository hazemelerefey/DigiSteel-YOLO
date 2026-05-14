# API

This page documents the key classes and functions that exist in the current snapshot.

## A2 — GhostConv (Weight-Sharing)

Source: [ghost_conv.py](../digisteel/modules/ghost_conv.py)

### `GhostModule`

Responsibility:

- Implements GhostNet-style feature generation:
  - Compute “primary” features via a standard convolution.
  - Generate additional “ghost” features via cheaper depthwise transformations.
  - Concatenate and slice to target output channels.

Key constructor params:

- `in_channels`, `out_channels`
- `kernel_size`, `stride`
- `ratio`: how many total output channels are produced relative to the learned “primary” channels
- `dw_size`: kernel size of the depthwise “cheap operation”

Input/Output:

- Input: `x` with shape `(N, C_in, H, W)`
- Output: shape `(N, C_out, H_out, W_out)`

### `GhostConv`

Responsibility:

- Drop-in convolution-like layer using `GhostModule`.
- Intended to replace standard conv blocks in a backbone.

Usage:

```python
import torch
from digisteel.modules import GhostConv

layer = GhostConv(in_channels=64, out_channels=128, kernel_size=3, stride=2)
x = torch.randn(2, 64, 32, 32)
y = layer(x)
print(y.shape)  # (2, 128, 16, 16)
```

### `GhostConvWeightSharing`

Responsibility:

- A2 novelty primitive: a single Ghost block reused across multiple pyramid stages.
- You instantiate it once, then apply it to multiple feature maps (P3/P4/P5) so parameters are shared.

Usage:

```python
import torch
from digisteel.modules.ghost_conv import GhostConvWeightSharing

shared = GhostConvWeightSharing(in_channels=64, out_channels=64)

p3 = torch.randn(1, 64, 80, 80)
p4 = torch.randn(1, 64, 40, 40)
p5 = torch.randn(1, 64, 20, 20)

out_p3 = shared(p3)
out_p4 = shared(p4)
out_p5 = shared(p5)
```

Notes:

- The caller is responsible for ensuring the feature maps have compatible channel dimensions.
- Spatial sizes can differ; the same module can operate on different `(H, W)` sizes.

## A3 — Inner-WIoU (Composite Regression Loss)

Source: [inner_wiou.py](../digisteel/modules/inner_wiou.py)

This loss is implemented on raw boxes and does not depend on a particular detector head.

Box format:

- All functions expect tensors shaped `(N, 4)` in `[x1, y1, x2, y2]` format.
- The caller should ensure `x2 >= x1` and `y2 >= y1` (or pre-process accordingly).

### `iou(box1, box2, eps=1e-7)`

Responsibility:

- Computes standard IoU for corresponding pairs of boxes.

Returns:

- Tensor of shape `(N,)`.

### `inner_iou_loss(pred_boxes, target_boxes, eps=1e-7)`

Responsibility:

- Computes an IoU-derived loss with an “inner rectangle” intuition.
- In the current implementation, the returned value is `mean(1 - IoU)`.

### `wiou_v3_loss(pred_boxes, target_boxes, eps=1e-7)`

Responsibility:

- Computes a Wise-IoU v3 inspired loss:
  - Base term: `1 - IoU`
  - Dynamic focusing weight based on aspect and scale discrepancy.

### `InnerWIoULoss(lambda_weight=0.5, eps=1e-7)`

Responsibility:

- Combines the two losses:

`L = λ · L_inner + (1 - λ) · L_wiou`

Usage:

```python
import torch
from digisteel.modules import InnerWIoULoss

loss_fn = InnerWIoULoss(lambda_weight=0.5)

pred = torch.tensor([[0.0, 0.0, 10.0, 10.0]])
target = torch.tensor([[1.0, 1.0, 11.0, 11.0]])

loss = loss_fn(pred, target)
print(loss.item())
```

## Public API Surface

The package re-exports these symbols:

- `from digisteel import GhostConv, InnerWIoULoss`

See: [digisteel/__init__.py](../digisteel/__init__.py)
