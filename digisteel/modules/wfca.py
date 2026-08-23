"""
Wavelet Frequency Channel Attention (WFCA) Module.

Novel contribution of DigiSteel-YOLO v2. Applies 2D Haar wavelet decomposition
to feature maps, computes cross-subband channel attention, and reconstructs
via inverse DWT — giving the network explicit frequency-domain reasoning.

Differentiators from prior work:
1. Cross-subband interaction (subbands inform each other's attention)
2. Per-subband attention BEFORE fusion (not concatenate-then-attend)
3. Inverse DWT reconstruction (maintains spatial resolution)

References:
    - Closest related work (must cite):
      MWYOLO (MDPI 2026): Multi-kernel convolutions on subbands, no inverse DWT
      YOLOv11-WBD (PLOS One 2025): Concatenates subbands then attends, no inverse DWT
      HDSA-YOLO (2025): Haar wavelet fusion + Swin attention, not per-subband channel attention
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def haar_dwt_2d(x: torch.Tensor):
    """
    2D Haar Discrete Wavelet Transform (parameter-free).

    Decomposes input into 4 subbands using separable Haar wavelets.
    H and W are padded to even if odd (reflect padding).

    Args:
        x: Input tensor (B, C, H, W).

    Returns:
        Tuple of (LL, LH, HL, HH), each (B, C, H', W') where H'≈H/2, W'≈W/2.
        LL = low-frequency approximation
        LH = horizontal detail (vertical edges)
        HL = vertical detail (horizontal edges)
        HH = diagonal detail (fine textures)
    """
    if x.shape[2] % 2 == 1:
        x = F.pad(x, (0, 0, 0, 1), mode="reflect")
    if x.shape[3] % 2 == 1:
        x = F.pad(x, (0, 1, 0, 0), mode="reflect")

    x_even_h = x[:, :, 0::2, :]
    x_odd_h = x[:, :, 1::2, :]

    l = (x_even_h + x_odd_h) * 0.5
    h = (x_even_h - x_odd_h) * 0.5

    ll = (l[:, :, :, 0::2] + l[:, :, :, 1::2]) * 0.5
    lh = (l[:, :, :, 0::2] - l[:, :, :, 1::2]) * 0.5
    hl = (h[:, :, :, 0::2] + h[:, :, :, 1::2]) * 0.5
    hh = (h[:, :, :, 0::2] - h[:, :, :, 1::2]) * 0.5

    return ll, lh, hl, hh


def haar_idwt_2d(ll: torch.Tensor, lh: torch.Tensor,
                  hl: torch.Tensor, hh: torch.Tensor):
    """
    2D Haar Inverse Discrete Wavelet Transform (parameter-free).

    Perfectly reconstructs the input from 4 subbands (within float precision).
    Assumes even spatial dimensions (matching the DWT output).

    Args:
        ll, lh, hl, hh: Subband tensors, each (B, C, H/2, W/2).

    Returns:
        Reconstructed tensor (B, C, H, W).
    """
    l_even = ll + lh
    l_odd = ll - lh
    h_even = hl + hh
    h_odd = hl - hh

    B, C, H2, W2 = ll.shape

    l = torch.zeros(B, C, H2, W2 * 2, device=ll.device, dtype=ll.dtype)
    l[:, :, :, 0::2] = l_even
    l[:, :, :, 1::2] = l_odd

    h = torch.zeros(B, C, H2, W2 * 2, device=ll.device, dtype=ll.dtype)
    h[:, :, :, 0::2] = h_even
    h[:, :, :, 1::2] = h_odd

    out = torch.zeros(B, C, H2 * 2, W2 * 2, device=ll.device, dtype=ll.dtype)
    out[:, :, 0::2, :] = l + h
    out[:, :, 1::2, :] = l - h

    return out


class WFCA(nn.Module):
    """
    Wavelet Frequency Channel Attention.

    Decomposes input features via Haar DWT, applies cross-subband channel
    attention to each subband independently, then reconstructs via inverse DWT.
    A learnable residual scaling factor (alpha) controls contribution.

    Architecture (per spec Section 2.2):
        Step 1: 2D Haar DWT -> [LL, LH, HL, HH] (or 7 subbands for 2-level)
        Step 2: GAP per subband -> concat -> shared FC context -> per-subband Sigmoid attention
        Step 3: Apply attention weights -> inverse DWT reconstruction
        Step 4: Residual: F_out = F + alpha * F_enhanced

    Args:
        channels: Number of input/output channels (C).
        reduction: Channel reduction ratio for the shared context FC layer.
        dwt_levels: Number of DWT decomposition levels (1 or 2).
            1-level: 4 subbands (LL, LH, HL, HH)
            2-level: 7 subbands (LL2, LH2, HL2, HH2, LH1, HL1, HH1)
    """

    def __init__(self, channels: int, reduction: int = 8, dwt_levels: int = 1):
        super().__init__()
        assert dwt_levels in (1, 2), "dwt_levels must be 1 or 2"
        self.channels = channels
        self.reduction = reduction
        self.dwt_levels = dwt_levels
        self._built = False

        # Learnable residual scaling via sigmoid (always nonzero gradient)
        self.alpha_raw = nn.Parameter(torch.tensor(-2.2))  # sigmoid(-2.2) ≈ 0.1

    def _build(self, actual_channels: int):
        """Lazily build linear layers based on actual input channels."""
        num_subbands = 4 if self.dwt_levels == 1 else 7
        self.channels = actual_channels

        # Cross-subband shared context
        self.shared_fc = nn.Sequential(
            nn.Linear(num_subbands * actual_channels, actual_channels // self.reduction),
            nn.ReLU(inplace=True),
        ).to(self.alpha_raw.device)

        # Per-subband attention projection (each subband gets its own FC)
        self.subband_fcs = nn.ModuleList([
            nn.Linear(actual_channels // self.reduction, actual_channels)
            for _ in range(num_subbands)
        ]).to(self.alpha_raw.device)

        self._built = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape

        # Lazy initialization on first forward pass
        if not self._built:
            self._build(C)

        # Step 1: Level 1 DWT
        ll, lh, hl, hh = haar_dwt_2d(x)

        if self.dwt_levels == 1:
            subbands = [ll, lh, hl, hh]
        else:
            # Step 1b: Level 2 — decompose LL further
            ll2, lh2, hl2, hh2 = haar_dwt_2d(ll)
            subbands = [ll2, lh2, hl2, hh2, lh, hl, hh]

        # Step 2: Global average pooling per subband -> descriptors
        descriptors = []
        for sb in subbands:
            desc = sb.mean(dim=(2, 3))  # (B, C)
            descriptors.append(desc)

        # Concatenate all descriptors: (B, num_subbands * C)
        concat_desc = torch.cat(descriptors, dim=1)

        # Shared context
        ctx = self.shared_fc(concat_desc)  # (B, C // r)

        # Per-subband attention weights
        attended_subbands = []
        for i, sb in enumerate(subbands):
            w = torch.sigmoid(self.subband_fcs[i](ctx))  # (B, C)
            w = w.unsqueeze(-1).unsqueeze(-1)  # (B, C, 1, 1)
            attended_subbands.append(w * sb)

        # Step 3: Inverse DWT reconstruction
        if self.dwt_levels == 1:
            enhanced = haar_idwt_2d(*attended_subbands)
        else:
            # Reconstruct level 2 first
            ll_recon = haar_idwt_2d(
                attended_subbands[0], attended_subbands[1],
                attended_subbands[2], attended_subbands[3],
            )
            # Level 2 IDWT may produce slightly different spatial size than
            # the level-1 subbands (due to padding odd dims at level 2).
            # Pad level-1 subbands to match ll_recon before final IDWT.
            rh, rw = ll_recon.shape[2], ll_recon.shape[3]
            lh1, hl1, hh1 = attended_subbands[4], attended_subbands[5], attended_subbands[6]
            if lh1.shape[2] != rh or lh1.shape[3] != rw:
                lh1 = F.pad(lh1, (0, max(0, rw - lh1.shape[3]), 0, max(0, rh - lh1.shape[2])))
                hl1 = F.pad(hl1, (0, max(0, rw - hl1.shape[3]), 0, max(0, rh - hl1.shape[2])))
                hh1 = F.pad(hh1, (0, max(0, rw - hh1.shape[3]), 0, max(0, rh - hh1.shape[2])))
            enhanced = haar_idwt_2d(ll_recon, lh1, hl1, hh1)

        # Crop to original size (in case padding was added for odd dims)
        enhanced = enhanced[:, :, :H, :W]

        # Step 4: Residual connection with learnable alpha (sigmoid for nonzero gradient)
        alpha = torch.sigmoid(self.alpha_raw)
        return x + alpha * enhanced
