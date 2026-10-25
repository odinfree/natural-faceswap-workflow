# Natural Face-Swap Workflow

A reusable pipeline for creating an authorized FaceFusion swap and then restoring the original actor's eye,
eyelid, brow, lip, teeth, and tongue performance. The result keeps the replacement identity while avoiding the
frozen-eye and synthetic-mouth look common in one-pass face swaps.

The repository contains code only. It intentionally excludes identity references, film clips, model files,
generated media, API keys, and platform credentials.

## Pipeline

```text
authorized identity references
            |
            v
FaceFusion base swap (HyperSwap by default)
            |
            v
MediaPipe landmarks + OpenCV expression transfer
  - eyes, eyelids, gaze and restrained brows
  - natural inner mouth, teeth, tongue and lip motion
            |
            v
FFmpeg H.264 master + contact sheet + optional Telegram GIF
```

HeyGen is optional. It can supply authorized reference images, but it does not perform the final swap in this
workflow.

## Safety and rights

Use this only when the replacement identity has explicitly authorized the use. You are also responsible for the
rights to the motion source and the selected FaceFusion model weights.

- Identity authorization is required for every run.
- Commercial mode is blocked unless source rights and model rights are both marked `confirmed`.
- A public URL is not proof that footage may be republished.
- Open-source FaceFusion code does not grant one blanket license for every bundled model.
- The tool creates local files only. It never uploads or publishes media.

This is an operational guardrail, not legal advice.

## Requirements

- Python 3.10-3.12
- [uv](https://docs.astral.sh/uv/)
- FFmpeg and FFprobe
- A separate [FaceFusion](https://github.com/facefusion/facefusion) checkout for the base swap
- CUDA is recommended for FaceFusion; expression repair is fast on CPU

On macOS:

```bash
brew install ffmpeg uv
```

## Install

```bash
git clone https://github.com/odinfree/natural-faceswap-workflow.git
cd natural-faceswap-workflow
uv sync --extra dev
uv run natural-faceswap download-model
```

The MediaPipe face-landmarker model is downloaded from Google's official model bucket into the ignored `models/`
directory.

## 1. Inspect the source

```bash
uv run natural-faceswap inspect inputs/source.mp4
```

Prefer a real 720p or 1080p video over a GIF repost. The command reports resolution, frame rate, duration, frame
count, codecs, and file size.

## 2. Create the FaceFusion base swap

Install FaceFusion separately, then ask this tool to print the exact command:

```bash
uv run natural-faceswap facefusion \
  --facefusion-home ../facefusion \
  --reference references/front.jpg references/three-quarter.jpg references/side.jpg \
  --target inputs/source.mp4 \
  --output outputs/base-swap.mp4 \
  --provider cuda \
  --usage internal-evaluation \
  --source-rights unknown \
  --model-rights unknown \
  --identity-authorized
```

Review the printed command. Add `--run` when the paths and rights state are correct. The default model is
`hyperswap_1a_256` and the recommended FaceFusion commit is recorded in the command output.

For a shot containing several similarly sized faces, add `--selector-mode reference` plus an inspected
`--reference-frame-number` and zero-based `--reference-face-position`. A selector that targets the wrong person is
not repaired by the later expression pass.

## 3. Restore natural expressions

The source and base swap must be frame-aligned and have the same dimensions and frame count.

```bash
uv run natural-faceswap transfer \
  --source inputs/source.mp4 \
  --swapped outputs/base-swap.mp4 \
  --output-dir outputs/final \
  --model models/face_landmarker.task \
  --expected-face 0.51,0.36 \
  --mode both \
  --gif \
  --caption-png inputs/caption-720x405.png \
  --usage internal-evaluation \
  --source-rights unknown \
  --model-rights unknown \
  --identity-authorized
```

`--expected-face` is the approximate normalized `x,y` center of the person to replace. This reduces wrong-person
tracking when multiple faces appear.

Process one coherent shot at a time when the target changes position across cuts. Join accepted shots afterward;
do not expect one normalized position to track a person through an edited sequence automatically.

Outputs:

- `final.mp4`: 1080p-style H.264 master at the source resolution and frame rate
- `telegram.gif`: optional 720x405 looping GIF
- `contact-sheet.jpg`: twelve samples across the finished clip
- `manifest.json`: hashes, base-model provenance, tool settings, rights state, inputs, outputs, and landmark failures

## Expression controls

The defaults reproduce the natural variant selected after side-by-side testing:

```text
eye core alpha     0.78
brow alpha         0.28
mouth core alpha   0.98
lip-ring alpha     0.48
```

The eye pass transfers the lids and iris strongly but keeps the surrounding skin restrained. The mouth pass uses
a strong inner-mouth blend and a softer outer-lip ring so facial hair and jaw identity stay intact.

Use `--mode eyes` or `--mode mouth` for a narrow correction. Change one setting at a time and compare contact
sheets before processing a larger batch.

## GPU and Runpod

Start FaceFusion on CUDA with an A10 24 GB or better. Keep expression repair and FFmpeg packaging local unless a
large batch justifies remote compute. Test CUDA before TensorRT; use TensorRT only after a representative sample
shows equal identity, masking, and expression quality. Stop the pod after downloading the verified result.

See [docs/WORKFLOW.md](docs/WORKFLOW.md) for the quality gates and [docs/RUNPOD.md](docs/RUNPOD.md) for a concise
remote checklist.

## Development

```bash
uv sync --extra dev
uv run ruff check .
uv run pytest
```

## License

The workflow code is MIT licensed. Media, identity references, FaceFusion, pretrained model weights, and external
services retain their own licenses and terms.
