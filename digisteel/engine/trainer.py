"""
DigiSteel-YOLO v2 Custom Trainer.

Approach: Subclass Ultralytics DetectionTrainer and override get_model() to inject
Inner-WIoU loss AFTER the model criterion is built. This is more reliable than
monkey-patching trainer.train() because get_model() is guaranteed to run before
any training step and the criterion is available at that point.

Components:
1. register_custom_modules() - injects GhostConv, WFCA, EMA into ultralytics namespace
2. InnerWIoUAdapter - replaces IoU computation inside BboxLoss with Inner-WIoU
3. DigiSteelTrainer - proper DetectionTrainer subclass with loss injection
4. patch_model_for_digisteel() - utility for inference model preparation

Usage:
    from digisteel.engine.trainer import DigiSteelTrainer, register_custom_modules

    register_custom_modules()
    trainer = DigiSteelTrainer(overrides={"data": "configs/data/neu_det.yaml", ...})
    trainer.train()
"""

import torch
from ultralytics.models.yolo.detect import DetectionTrainer


def register_custom_modules():
    """
    Register custom modules in the Ultralytics namespace.

    Must be called BEFORE loading any YAML config that references these modules.
    This injects custom modules into ultralytics.nn.tasks globals
    so the YAML parser can find them by name.
    """
    import ultralytics.nn.tasks as tasks
    from digisteel.modules import CoordAttention, DAFE, EMA, GhostConv, WFCA

    tasks.CoordAttention = CoordAttention
    tasks.DAFE = DAFE
    tasks.GhostConv = GhostConv
    tasks.WFCA = WFCA
    tasks.EMA = EMA


class InnerWIoUAdapter:
    """
    Adapter that computes Inner-WIoU IoU values for Ultralytics BboxLoss.

    Ultralytics' BboxLoss.forward() computes iou = self.iou(pred_bboxes, target_bboxes)
    and feeds the result into the loss. This adapter replaces that iou() call with
    our Inner-WIoU computation while keeping the same interface.

    The key insight: we don't replace the entire BboxLoss — we just replace the
    IoU computation method. This is more robust across Ultralytics versions.
    """

    def __init__(self, lambda_weight: float = 0.5):
        self.lambda_weight = lambda_weight

    def __call__(self, pred_bboxes, target_bboxes, eps=1e-7):
        """Compute Inner-WIoU IoU values, matching standard iou() interface."""
        from digisteel.modules.inner_wiou import iou as basic_iou

        # Standard IoU
        iou_val = basic_iou(pred_bboxes, target_bboxes, eps)

        # WIoU-style aspect/scale focus
        pred_w = pred_bboxes[:, 2] - pred_bboxes[:, 0]
        pred_h = pred_bboxes[:, 3] - pred_bboxes[:, 1]
        tgt_w = target_bboxes[:, 2] - target_bboxes[:, 0]
        tgt_h = target_bboxes[:, 3] - target_bboxes[:, 1]

        pw = pred_w.clamp(min=eps)
        ph = pred_h.clamp(min=eps)
        tw = tgt_w.clamp(min=eps)
        th = tgt_h.clamp(min=eps)

        aspect_diff = torch.abs(pw / ph - tw / th)
        scale_diff = torch.abs(torch.sqrt(pw * ph) - torch.sqrt(tw * th))
        focus = torch.exp(-(aspect_diff + scale_diff))

        return iou_val * self.lambda_weight + iou_val * focus * (1 - self.lambda_weight)


class DigiSteelTrainer(DetectionTrainer):
    """
    Custom trainer for DigiSteel-YOLO v2.

    Subclasses DetectionTrainer to inject Inner-WIoU loss into BboxLoss.
    Two injection points ensure reliability across Ultralytics versions:
    1. get_model() — patches immediately after model build
    2. _do_train() — second-chance patch before first training step
    """

    def get_model(self, cfg=None, weights=None, verbose=True):
        """Build model and inject Inner-WIoU into BboxLoss."""
        self._iou_patched = False
        model = super().get_model(cfg, weights, verbose)
        self._try_patch_loss(model, verbose)
        return model

    def _do_train(self):
        """Second-chance loss injection before training starts."""
        if not getattr(self, "_iou_patched", False) and hasattr(self, "model"):
            self._try_patch_loss(self.model, verbose=True)
        return super()._do_train()

    def _try_patch_loss(self, model, verbose=False):
        """Attempt to inject InnerWIoUAdapter into bbox_loss.iou."""
        if getattr(self, "_iou_patched", False):
            return
        if hasattr(model, "criterion") and hasattr(model.criterion, "bbox_loss"):
            model.criterion.bbox_loss.iou = InnerWIoUAdapter(lambda_weight=0.5)
            self._iou_patched = True
            if verbose:
                print("  [DigiSteel] Inner-WIoU loss injected into BboxLoss")
        elif verbose:
            print("  [DigiSteel] Warning: criterion.bbox_loss not yet available")

    @classmethod
    def create_trainer(cls, overrides=None):
        """Factory method (backwards compatible with older notebook code)."""
        register_custom_modules()
        return cls(overrides=overrides or {})


def patch_model_for_digisteel(model):
    """
    Patch a YOLO model for DigiSteel inference.

    Registers custom modules and prepares the model for use with
    DigiSteel v2 architecture YAML.

    Args:
        model: YOLO model instance.

    Returns:
        The patched model (same instance).
    """
    register_custom_modules()

    if hasattr(model, "model") and model.model is not None:
        info = model.model
        num_params = sum(p.numel() for p in info.parameters()) if hasattr(info, "parameters") else 0
        print("  [DigiSteel] Model loaded: {:.2f}M params".format(num_params / 1e6))

        # Check for expected modules
        from digisteel.modules.wfca import WFCA
        from digisteel.modules.ema import EMA
        from digisteel.modules.ghost_conv import GhostConv

        wfca_count = sum(1 for m in info.modules() if isinstance(m, WFCA))
        ema_count = sum(1 for m in info.modules() if isinstance(m, EMA))
        gc_count = sum(1 for m in info.modules() if isinstance(m, GhostConv))

        print("  [DigiSteel] Modules: {} GhostConv, {} WFCA, {} EMA".format(gc_count, wfca_count, ema_count))

    return model
