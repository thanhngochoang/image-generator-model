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

Face stickers for the Lumen camera, **cut from real photos** that are CC0 or public domain (Openverse search limited to
`cc0,pdm`; each photo's title, author, source page and licence are in `stickers/pack/CREDITS.json` and
`stickers/photos/src/credits.json`). The pack itself is released under **CC0-1.0**.

| File | Contents |
|---|---|
| `stickers/pack/face_stickers.zip` | `stickers.json` + 9 PNG pictures (stored, not compressed) |
| `stickers/pack/stickers.json` | 7 stickers: cat (sphynx ears + kitten nose), bunny (rabbit ears), puppy (boxer ears + dog nose), tiara (crystal tiara), sunglasses (yellow), flower crown (real flowers set along an arc), blush |

Each sticker lists its parts: a picture, the face **anchor** it sits on (`forehead`, `eyes`, `nose`, `mouth`, `chin`, `cheek_left`, `cheek_right`, from MediaPipe Face Landmarker points 10, 33/263, 1, 13/14, 152, 50, 280), its **width** in face widths (points 234 – 454) and an offset **dx / dy** in face widths along the face's own axes (dy < 0 = up). The app turns each part with the line between the outer eye corners.

Download (release `stickers-2026.11.25`): `face_stickers.zip`, SHA-256 `3ba09dbea2bed77280323bacd421a04f47fa0a4cc2bd998cf559f02d47e2f10a`, 3 226 454 bytes. (The earlier drawn pack, release `stickers-2026.11.23`, stays available for older app versions.)

Rebuild: `python stickers/photos/make_real_stickers.py stickers/photos/src <isnet-general-use.onnx> stickers/pack` (Python 3.12, `onnxruntime`, `numpy`, `pillow`). Cutouts use ISNet / DIS "general use" (Apache-2.0, the `isnet-general-use.onnx` file of the rembg project), run only to make the pictures; each part is limited by a hand-drawn region (an ear, a nose). `photos/preview.py` places every sticker on a face with the app's own maths for checking; `photos/search.py` and `photos/fetch.py` are the search and download used to pick the photos.

## Beat tracking (`beat/`)

`beat/beat_this_small0_int8w.onnx` (3.5 MB, SHA-256 `30816a35c7bde6b43186d492a150d03677262e4b96c3381c0629e2a510b32e9d`, 3 549 603 bytes): **Beat This!** `small0` (Foscarin, Schlüter & Widmer, "Beat this! Accurate beat tracking without DBN postprocessing", ISMIR 2024, CP JKU Linz) for the Lumen music slideshow.

- **Source:** the official `small0` checkpoint (`small0.ckpt`, SHA-256 `6074be2c4d490c5f6101fcc374a1ec72ae93456e23bb6019783b849f5dc7d47b`) from github.com/CPJKU/beat_this.
- **Export:** ONNX opset 17 by `beat/src/convert.py`. Input `spect` float32 [1, frames, 128] (log-mel, 22 050 Hz, hop 441, 50 frames/s); outputs `beat` and `downbeat` logits [1, frames]. The fp32 export is within 4.7e-5 logits of PyTorch.
- **Int8:** weight-only, per output channel (`scripts/quantize.py`), 10.5 MB → 3.5 MB; all arithmetic stays fp32.
- **Quality vs the fp32 model** (pass threshold set before measuring: beats and downbeats within ±20 ms with F-measure ≥ 0.98, same BPM, on every piece):
  - 10 one-minute excerpts of real (orchestral / cinematic) music: beats F ≥ 0.983 on every excerpt; same BPM on 9 of 10 (one excerpt with an ambiguous pulse: 176 vs 167); downbeats identical on 8 of 10, one downbeat differs on each of the other two (F 0.977).
  - 7 synthetic pieces (`beat/src/quant_eval.py`): beats F ≥ 0.981; downbeats below 0.98 on 2 pieces the fp32 model itself mis-tracks. Against the true beats of those pieces int8 scores the same as fp32 (within 0.01).
  - The strict threshold is not met (single downbeats on 2 excerpts). It was accepted because beat timing, which drives the slideshow cuts, matches. In Lumen on an Android emulator, a 120 BPM test export is identical to the fp32 one.
- **Licence:** MIT (code and checkpoints, https://github.com/CPJKU/beat_this/blob/main/LICENSE). Changes: ONNX export and int8 weight re-encoding, no retraining. Scripts in `beat/src/`: MIT.
- **Reproduce:** `beat/src/convert.py` (export), `beat/src/quant_eval.py <scripts dir> wo` (quantize + synthetic gate), `beat/src/real_eval.py <scripts dir> <folder of mp3>` (real-music gate). Python 3.12 with `torch`, `torchaudio`, `onnx`, `onnxruntime==1.32.0`, `numpy`, `pillow`, `av`, and `beat-this` from github.com/CPJKU/beat_this.

Download (release `beat-2026.11.23`): `beat_this_small0_int8w.onnx`.
