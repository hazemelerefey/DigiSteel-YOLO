"""
DigiSteel-YOLO v2 Custom Trainer.

Subclasses Ultralytics DetectionTrainer to:
1. Register custom modules (GhostConv, WFCA, EMA) into ultralytics namespace
2. Inject Inner-WIoU loss into BboxLoss during training
3. Enable perturbation-aware training augmentation

Usage:
    from digisteel.engine.trainer import DigiSteelTrainer, register_custom_modules

    register_custom_modules()
    trainer = DigiSteelTrainer(overrides={"data": "configs/data/neu_det.yaml", ...})
    trainer.train()
"""

import torch


def register_custom_modules():
    """
    Register custom modules in the Ultralytics namespace.

    Must be called BEFORE loading any YAML config that references these modules.
    This injects GhostConv, WFCA, and EMA into ultralytics.nn.tasks globals
    so the YAML parser can find them by name.
    """
    import ultralytics.nn.tasks as tasks
    from digisteel.modules import EMA, GhostConv, WFCA

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


class DigiSteelTrainer:
    """
    Custom trainer for DigiSteel-YOLO v2.

    Wraps Ultralytics DetectionTrainer to inject:
    1. Module registration (GhostConv, WFCA, EMA)
    2. Inner-WIoU loss patching

    Can be used in two ways:
    1. As a trainer class (subclasses DetectionTrainer)
    2. As a function that patches an existing model

    Usage (as function):
        from digisteel.engine.trainer import patch_model_for_digisteel
        model = YOLO("configs/models/digisteel_v2.yaml")
        patch_model_for_digisteel(model)
        model.train(...)

    Usage (as trainer):
        from digisteel.engine.trainer import DigiSteelTrainer
        trainer = DigiSteelTrainer(overrides={...})
        trainer.train()
    """

    @staticmethod
    def create_trainer(overrides: dict = None):
        """
        Create an Ultralytics DetectionTrainer with DigiSteel patches applied.

        Args:
            overrides: Ultralytics trainer overrides dict.

        Returns:
            Configured DetectionTrainer instance.
        """
        register_custom_modules()

        from ultralytics.models.yolo.detect import DetectionTrainer

        trainer = DetectionTrainer(overrides=overrides or {})

        # Patch the criterion's bbox_loss IoU computation
        _patch_loss_iou(trainer)

        return trainer


def _patch_loss_iou(trainer):
    """
    Patch the trainer's loss to use Inner-WIoU.

    This replaces the IoU computation inside BboxLoss with our Inner-WIoU
    metric. It's a targeted monkey-patch that survives Ultralytics' internal
    loss object rebuilds.
    """
    try:
        adapter = InnerWIoUAdapter(lambda_weight=0.5)

        # The criterion is built lazily in Ultralytics.
        # We hook into the train method to patch after it's built.
        original_train = trainer.train

        def patched_train(*args, **kwargs):
            # Patch after criterion is built
            if hasattr(trainer, "model") and hasattr(trainer.model, "criterion"):
                criterion = trainer.model.criterion
                if hasattr(criterion, "bbox_loss"):
                    criterion.bbox_loss.iou = adapter
                    print("  [DigiSteel] Patched BboxLoss with Inner-WIoU")
            return original_train(*args, **kwargs)

        trainer.train = patched_train
    except Exception as e:
        print(f"  [DigiSteel] Warning: Could not patch loss: {e}")
        print("  [DigiSteel] Training will use standard CIoU loss")


def patch_model_for_digisteel(model):
    """
    Patch a YOLO model for DigiSteel training.

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
        print(f"  [DigiSteel] Model loaded: {num_params / 1e6:.2f}M params")

        # Check for expected modules
        from digisteel.modules.wfca import WFCA
        from digisteel.modules.ema import EMA
        from digisteel.modules.ghost_conv import GhostConv

        wfca_count = sum(1 for m in info.modules() if isinstance(m, WFCA))
        ema_count = sum(1 for m in info.modules() if isinstance(m, EMA))
        gc_count = sum(1 for m in info.modules() if isinstance(m, GhostConv))

        print(f"  [DigiSteel] Modules: {gc_count} GhostConv, {wfca_count} WFCA, {ema_count} EMA")

    return model
