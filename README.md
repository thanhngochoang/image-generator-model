# image-generator-model

Smaller (int8-weight) ONNX models for the Lumen offline photo editor. Each file is a drop-in replacement for the original:
same input and output names, shapes and types, standard ONNX ops, tested with ONNX Runtime 1.32.0 (desktop and Android).

## Files

| File | Replaces | Original size | This file | SHA-256 |
|---|---|---|---|---|
| `gfpgan/GFPGANv1.4_int8w.onnx` | GFPGAN v1.4 (face restoration) | 340.4 MB | 85.5 MB | `e482f34e97745f38c680fe8b315d81cce57067b2f700829411a71f7be0cb0665` |
| `ddcolor/ddcolor_tiny_int8w.onnx` | DDColor tiny (colourisation) | 135.4 MB (fp16) | 68.8 MB | `1115e67387c7a2134594ce7d786d466c9b9b5fc6d301ae1413651475c0df90c2` |
| `mobile_sam/mobile_sam_encoder_int8w.onnx` | MobileSAM image encoder | 28.2 MB | 8.4 MB | `64ede3fbdfc1011cf7463ce6911c5b9de8e8e16c0df8af0d449b61b58a2c976a` |
| `mobile_sam/mobile_sam_decoder_int8w.onnx` | MobileSAM prompt decoder | 16.5 MB | 4.9 MB | `13a8e755cd3a0394d2f5a4d5c6c7943fc809ac233e189de70761dbc67359ffbc` |

Use the two MobileSAM files together (the encoder was validated with the int8 decoder). `mobile_sam/mobile_sam_int8w.zip` holds both (stored, not compressed) as `mobile_sam_encoder.onnx` and `mobile_sam_decoder.onnx`, for apps that download one file per pack.

## Sources (originals)

- GFPGAN v1.4 ONNX: https://huggingface.co/HowToSD/GFPGAN-ONNX/resolve/ae7b761a0dd6da7ccd7d1f02cc584a2f1df39620/GFPGANv1.4.onnx (from TencentARC/GFPGAN)
- DDColor tiny: https://huggingface.co/edgetools/ddcolor/resolve/4755ae9f1f7a35a9e7693b96c2a88f3432cb6ab0/ddcolor-tiny-fp16.onnx (weights from piddnad/ddcolor_paper_tiny). It was first up-cast to fp32, then quantized.
- MobileSAM: https://huggingface.co/vietanhdev/segment-anything-onnx-models/resolve/9effc01a9e135621d710d49159f1ffb0b6f724dc/mobile_sam_20230629.zip (from ChaoningZhang/MobileSAM)

## Method

Weight-only int8: weights are stored as int8 with fp32 scales (per output channel for GFPGAN and DDColor, one scale per block of 32 weights for MobileSAM) and turned back into fp32 by `DequantizeLinear` when the model loads. All arithmetic stays fp32.

- Download size is about 4× smaller. Run-time memory is NOT smaller: weights are held as fp32 after loading.
- Speed is about the same as fp32 on CPU.
- DDColor: the original fp16 release gives over-saturated colours on ONNX Runtime Android (fp16 arithmetic); this file computes in fp32 and matches the fp32 reference.
- ONNX Runtime's dynamic quantization including Conv, and static QDQ quantization, were tried and rejected: they broke these networks (see the study notes in `scripts/`).

## Quality (vs the original, worst input of 5–7 per model)

Pass thresholds were set before measuring: images PSNR ≥ 35 dB or mean abs diff ≤ 2/255; masks IoU ≥ 0.98; DDColor mean |Δab| ≤ 2 (Lab units).

| Model | Result | Verdict |
|---|---|---|
| GFPGAN | face PSNR ≥ 49.2 dB, mean abs diff ≤ 0.59/255 | pass |
| DDColor | mean \|Δab\| ≤ 1.64 vs fp32; on Android 1.1 vs the fp32 desktop output | pass |
| MobileSAM encoder + decoder | mask IoU ≥ 0.985 (encoder), ≥ 0.987 (decoder), ≥ 0.990 together | pass |
| MI-GAN (not included) | hole PSNR 29.4 dB | fail, keep the original |
| MODNet (not included) | IoU 0.9988 but only 9 % smaller | not worth it |

Also checked inside the Lumen app on an Android emulator: face restoration, colourisation and tap-to-select run correctly with these files.

## Reproduce

`scripts/quantize.py` (weight-only quantization), `scripts/evaluate.py` (metrics), `scripts/feeds.py` (pre-processing that mirrors the app), `scripts/make_report.py`. Paths come from environment variables: `MODEL_PACKS` (original models), `TEST_IMAGES` (test photos), `INT8_ROOT` (output). Requires Python 3.12, `onnx`, `onnxruntime==1.32.0`, `numpy`, `pillow`.

## Licences

These files are derivatives of the originals and keep their licences.

- DDColor: Apache-2.0 (https://github.com/piddnad/DDColor/blob/master/LICENSE).
- MobileSAM: Apache-2.0 (https://github.com/ChaoningZhang/MobileSAM/blob/master/LICENSE).
- GFPGAN: Apache-2.0 for GFPGAN itself, plus components under non-commercial licences (StyleGAN2 from NVIDIA, DFDNet). See `gfpgan/LICENSE-GFPGAN.md`. Use the GFPGAN file for non-commercial purposes only.
- The scripts in `scripts/`: Apache-2.0.

Changes made: weights re-encoded as int8 with per-channel or per-block scales (and, for DDColor, up-cast from fp16 to fp32 before that). No retraining.

## Face stickers (`stickers/`)

Original face stickers for the Lumen camera, drawn by `stickers/src/make_stickers.py` (Pillow, no outside images). Released under **CC0-1.0**: free for any use, no attribution needed.

| File | Contents |
|---|---|
| `stickers/pack/face_stickers.zip` | `stickers.json` + 15 PNG pictures (stored, not compressed) |
| `stickers/pack/stickers.json` | 12 stickers: cat, bunny, crown, heart glasses, sunglasses, blush, flower crown, mustache, party hat, devil horns, halo, puppy |

Each sticker lists its parts: a picture, the face **anchor** it sits on (`forehead`, `eyes`, `nose`, `mouth`, `chin`, `cheek_left`, `cheek_right`, from MediaPipe Face Landmarker points 10, 33/263, 1, 13/14, 152, 50, 280), its **width** in face widths (points 234 – 454) and an offset **dx / dy** in face widths along the face's own axes (dy < 0 = up). The app turns each part with the line between the outer eye corners.

Download (release `stickers-2026.11.23`): `face_stickers.zip`, SHA-256 `6287b361962fbeb311226c745e336cf73161b8e845eb64bbfca416d431d4f030`, 388 343 bytes.

Rebuild: `python stickers/src/make_stickers.py stickers/pack` (Python 3.12, Pillow).

