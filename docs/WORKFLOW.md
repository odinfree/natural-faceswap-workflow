# Workflow and quality gates

## 1. Source gate

- Preserve the original file.
- Prefer native 1280x720 or better.
- Treat dimensions as reported resolution, not proof of native detail.
- Prefer video over GIF because GIFs discard color and often repeat frames.
- Record source URL, rights state, duration, frame rate, codecs, streams, and resolution.

## 2. Identity-reference gate

- Use only authorized identity references.
- Prefer front, three-quarter, and mild side views.
- Keep age, facial hair, and lighting reasonably consistent.
- Avoid sunglasses, extreme filters, hard shadows, or hands covering the face.

## 3. Base-swap gate

- Test a short representative sample before a full render.
- Include a frontal frame, strongest turn, fastest motion, largest occlusion, and difficult lighting.
- Change one variable at a time: model, selector, references, mask, or pixel boost.
- Preserve source timing and audio.

## 4. Expression-repair gate

- Confirm source and swap have identical dimensions and frame count.
- Set the expected face center to avoid tracking another person.
- Start with the default eye and mouth alphas.
- If facial hair weakens, lower `--lip-alpha` before changing the mouth core.
- If eyes look pasted, lower `--brow-alpha` before changing the eye core.

## 5. Full-frame QA

Reject the render for:

- wrong-person swaps or face dropout
- identity changing between frames
- frozen gaze or eyelids
- synthetic teeth, tongue, or mouth geometry
- hairline, moustache, glasses, teeth, or jaw flicker
- pasted mask edges or skin-tone seams
- hands or objects painted over
- missing action, timing drift, lost audio, or unexpected format changes

Inspect the beginning, middle, end, fastest motion, strongest turn, and every frame of the hardest expression.
An encode completing successfully is not visual approval.

## 6. Publication boundary

The CLI does not upload anything. Treat distribution as a separate decision after source rights, identity consent,
model rights, output QA, and platform rules have been checked.
