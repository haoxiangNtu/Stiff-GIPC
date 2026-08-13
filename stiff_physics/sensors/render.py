"""RGB tactile image rendering: reference image + learned normal->RGB residual.

Verbatim port of Taccel's ``VBTSMLP`` (taccel/vbts.py) so that the published
checkpoint ``taccel/data/n2rgb.pth`` loads unchanged, plus the compositing step
from ``TaccelModel.render_tactile``:

    rgb = ref_img + MLP(normal, pos_enc(pixel_coord))

The checkpoint was trained with in_len=9 (3 normal + 6 positional encoding,
n_pe=2) and img_h/img_w defaults of 512 -- keep those, they are baked into the
positional encoding.
"""

from __future__ import annotations

import numpy as np

try:  # torch is only needed for the RGB path
    import torch
    import torch.nn as nn

    _TORCH = True
except ImportError:  # pragma: no cover
    _TORCH = False
    nn = None

_N2RGB_IN_LEN = 9
_N2RGB_N_PE = 2


if _TORCH:

    class VBTSMLP(nn.Module):
        """Taccel's normal->RGB residual MLP (unchanged, so n2rgb.pth loads)."""

        def __init__(self, in_len=14, out_len=3, img_h=512, img_w=512, n_pe=4, input_depth=False):
            super().__init__()
            self.fc1 = nn.Sequential(
                nn.Linear(in_len, 256), nn.ReLU(),
                nn.Linear(256, 256), nn.ReLU(),
                nn.Linear(256, 256), nn.ReLU(),
            )
            self.fc2 = nn.Sequential(
                nn.Linear(in_len + 256, 256), nn.ReLU(),
                nn.Linear(256, 256), nn.ReLU(),
                nn.Linear(256, 256), nn.ReLU(),
                nn.Linear(256, out_len),
            )
            self.input_depth = input_depth
            self.img_h, self.img_w = img_h, img_w
            self.n_pe = n_pe
            self.x_pe_fn = [(lambda x, i=i: torch.sin(x * 2**i * torch.pi / img_h)) for i in range(n_pe)]
            self.y_pe_fn = [(lambda x, i=i: torch.cos(x * 2**i * torch.pi / img_w)) for i in range(n_pe)]

        def pos_enc(self, coord):
            x, y = coord[..., 0], coord[..., 1]
            return torch.concat(
                [
                    torch.stack([x] + [fn(x) for fn in self.x_pe_fn], dim=-1),
                    torch.stack([y] + [fn(y) for fn in self.y_pe_fn], dim=-1),
                ],
                dim=-1,
            )

        def forward(self, depth, normal, coord):
            x = (
                torch.concat([depth, normal, self.pos_enc(coord)], dim=-1)
                if self.input_depth
                else torch.concat([normal, self.pos_enc(coord)], dim=-1)
            )
            y = self.fc1(x)
            y = self.fc2(torch.concat([x, y], dim=-1))
            return torch.clip(y, -1.0, 1.0)


class TactileRenderer:
    """normal map -> RGB tactile image, using Taccel's published checkpoint."""

    def __init__(self, ckpt_path: str, ref_img_path: str, device: str = "cuda:0",
                 chunk_px: int = 1 << 20, half: bool = False,
                 flat_calibrate: bool = True):
        """
        Args:
            chunk_px: pixels evaluated per MLP forward. The MLP runs on EVERY
                pixel through 6x256 layers, so activations are
                ~3 * chunk_px * 256 * 4 B; a full 64-env batch at 400x400
                (10.2M px) would need ~10 GiB in one shot. 1<<20 keeps the
                render under ~1 GiB regardless of batch size.
            half: run the MLP in fp16 (halves activation memory and is ~1.5x
                faster; residual differences are ~1e-3 in [0,1] RGB).
            flat_calibrate: subtract the MLP's output for a flat surface
                (n = +z) so an untouched sensor renders EXACTLY the reference
                image.  Without it the checkpoint leaves a fixed spatial bias
                (mean 0.019, max 0.129 in [0,1] RGB) on the rest state, which
                would show up as a constant SSIM penalty against real images.
                Set False to reproduce Taccel's compositing literally.
        """
        if not _TORCH:
            raise ImportError("RGB rendering needs torch (pip install torch)")
        self.device = device
        self.chunk_px = int(chunk_px)
        self.dtype = torch.float16 if half else torch.float32
        self.mlp = VBTSMLP(in_len=_N2RGB_IN_LEN, out_len=3, n_pe=_N2RGB_N_PE)
        self.mlp.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
        self.mlp.to(device=device, dtype=self.dtype).eval()
        self.ref = self._load_ref(ref_img_path)
        h, w = self.ref.shape[:2]
        coords = np.stack(np.meshgrid(np.arange(h), np.arange(w)), axis=-1).transpose(1, 0, 2)
        self.coords = torch.from_numpy(coords).float().reshape(1, -1, 2).to(device)

        self.flat_bias = None
        if flat_calibrate:
            flat = np.zeros((1, h, w, 3), dtype=np.float32)
            flat[..., 2] = 1.0
            self.flat_bias = self._forward(np.zeros((1, h, w), np.float32), flat)

    @staticmethod
    def _load_ref(path: str) -> np.ndarray:
        try:
            import cv2

            img = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
        except ImportError:
            from PIL import Image

            img = np.asarray(Image.open(path).convert("RGB"))
        return img.astype(np.float64) / 255.0

    @property
    def resolution(self) -> tuple[int, int]:
        return self.ref.shape[:2]

    def render(self, depth: np.ndarray, normal: np.ndarray) -> np.ndarray:
        """(B, H, W) depth + (B, H, W, 3) normal -> (B, H, W, 3) RGB in [0, 1].

        H, W must match the reference image (400x400 for Taccel's 6k_ref.png).
        """
        residual = self._forward(depth, normal)
        if self.flat_bias is not None:
            residual = residual - self.flat_bias
        return np.clip(self.ref[None] + residual, 0.0, 1.0).astype(np.float32)

    def _forward(self, depth: np.ndarray, normal: np.ndarray) -> np.ndarray:
        """Raw MLP residual, (B, H, W, 3) float32 -- no reference image added."""
        depth = np.asarray(depth).reshape(-1, *self.resolution)
        b = depth.shape[0]
        npix = self.resolution[0] * self.resolution[1]
        normal = np.asarray(normal).reshape(b, npix, 3)

        with torch.no_grad():
            d = torch.from_numpy(np.ascontiguousarray(depth.reshape(b * npix, 1))).to(
                self.device, self.dtype)
            n = torch.from_numpy(np.ascontiguousarray(normal.reshape(b * npix, 3))).to(
                self.device, self.dtype)
            c = self.coords.reshape(1, npix, 2).expand(b, npix, 2).reshape(b * npix, 2).to(self.dtype)

            out = torch.empty((b * npix, 3), device=self.device, dtype=self.dtype)
            for lo in range(0, b * npix, self.chunk_px):
                hi = min(lo + self.chunk_px, b * npix)
                out[lo:hi] = self.mlp(d[lo:hi], n[lo:hi], c[lo:hi])

            return out.reshape(b, *self.resolution, 3).float().cpu().numpy()


def draw_markers(rgb: np.ndarray, markers_px: np.ndarray, flow_px: np.ndarray | None = None,
                 radius: int = 5) -> np.ndarray:
    """Overlay marker dots (and optional 4x flow arrows) on one (H, W, 3) image."""
    import cv2

    out = rgb.copy()
    for i, m in enumerate(markers_px):
        out = cv2.circle(out, m.astype(int), radius, (0.0, 0.0, 0.0), -1)
        if flow_px is not None:
            out = cv2.arrowedLine(
                out, (m - flow_px[i]).astype(int), (m + 4 * flow_px[i]).astype(int),
                color=(1.0, 0.0, 1.0), thickness=2, line_type=cv2.LINE_AA, tipLength=0.3,
            )
    return out
