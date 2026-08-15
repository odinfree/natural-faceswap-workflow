# Runpod checklist

Use Runpod for the FaceFusion base swap when CUDA materially reduces render time. The MediaPipe/OpenCV expression
repair usually runs quickly on a local CPU.

## Recommended starting point

- CUDA on an A10 24 GB or better
- persistent volume only if repeated model downloads justify it
- no API keys or identity references baked into an image
- no TensorRT until a representative CUDA comparison passes visual QA

## Session

1. Start the pod.
2. Clone FaceFusion and this repository.
3. Check out the FaceFusion commit printed by `natural-faceswap facefusion`.
4. Upload the motion source and authorized references through a private transfer path.
5. Run a short sample.
6. Inspect identity, occlusion, gaze, mouth, and frame timing.
7. Run the full base swap only after the sample passes.
8. Download the result and manifest.
9. Stop or delete the pod and confirm it is no longer billable.

Do not store cloud credentials, service keys, or third-party media in this Git repository.
