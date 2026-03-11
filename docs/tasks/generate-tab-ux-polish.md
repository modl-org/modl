# Generate Tab — UX Polish Tasks

Feedback from design review + code review findings. Ordered by impact.

## Code Health (from code review)

- [x] **Extract `submitJob()` helper** — handleGenerate and handleEdit share ~100 lines of identical logic (seed randomization, session item creation, SSE, error handling, toasts). Dedupe into a single function that takes request-building as a param.
- [x] **Fix re-render cascade** — `handleCancelQueued` depends on `[sessionItems]`, re-creating on every progress tick and cascading re-renders to SessionStrip + GenerateActions. Use a ref for sessionItems in callbacks. Same issue with `handleGenerate`/`handleEdit` depending on `[form]` (re-created on every keystroke).
- [x] **Close SSE on completion** — `setSseConnected(false)` is never called on successful generation. EventSource leaks an open connection indefinitely after first generation.
- [x] **Fix keyboard shortcut useEffect** — Missing dependency array on Ctrl+Enter handler (line 291). Re-registers listener every render.
- [x] **Stop mutating `form.seed` directly** — Lines 316, 426 bypass React immutability. Use the `newSeed` variable in request construction instead.
- [x] **Move keyframe animations to CSS** — `@keyframes gradientShift` and `shimmer` are in an inline `<style>` tag inside ImagePreview. Move to index.css.
- [x] **Fix collapsed training indicator position** — AppSidebar line 123: `absolute` dot inside button without `relative`. Dot renders in wrong position.
- [x] **Extract `useGpuStatus()` hook** — GPU query duplicated in App.tsx, AppSidebar.tsx, GenerateView.tsx with identical config.
- [x] **Guard division by zero** — GenerateProgressBar.tsx line 58: `totalSteps` could be 0, producing `Infinity%` CSS width.

## Prompt Area

- [x] **Move Enhance button inline** — Currently reads as a section toggle. Make it a pill/icon inside or attached to the bottom-right of the textarea so it reads as an action on the prompt.
- [x] **Auto-grow textarea** — Default is too tall for typical 1-3 line prompts. Start small, grow on content. Saves vertical sidebar space.
- [x] **Improve negative prompt toggle visibility** — Chevron is too subtle. Use a +/- icon or slightly bolder toggle for power user discoverability.

## Model & LoRA

- [x] **Split model info into two lines** — "Qwen Image 20B (gguf-q5km) 13.9GB" is too dense inline. Put quantization + size on a secondary muted line.
- [x] **Remove "No LoRAs applied" dead text** — The "+ Add" button is self-explanatory. Either hide the empty state entirely or make it an interactive drop zone.

## Dimensions

- [x] **Increase selected state contrast** — Purple outline on dark bg is low contrast. Use a filled background, not just border, for the active ratio preset.
- [x] **Widen hit targets** — Aspect ratio buttons are tight. Add more spacing between them.
- [x] **Visually tie dimensions to selected preset** — "1152 × 896 px" feels orphaned. Show it directly below the highlighted option or as a tooltip. But allow editing.

## Generation Controls

- [x] **Add tooltip/label to stop button** — Red square next to Enqueue has no affordance. Users can't tell if it cancels queue or stops current gen. Add tooltip at minimum.
- [x] **Expand queue status into mini drawer** — Truncated prompt text is hard to distinguish between jobs. The expandable queue panel exists but the collapsed state is too compressed. Consider always showing it as a mini-list when >1 job.
- [x] **Add GPU warning state** — "GPU: 7.6 GB free" is passive. When free VRAM is too low for the selected model, show a warning (yellow/red) instead of just static text.

## Generation Preview (Right Canvas)

- [x] **Show progress in loading state** — The circular progress ring exists but needs to be more visible.
- [x] **Differentiate prompt echo styling** — Prompt text below the loading card uses similar styling to the editable prompt. Use italics, quotes, or smaller/dimmer text so it reads as reference not input.
- [x] **Clarify X button behavior** — The close button on the generating card is ambiguous (cancel? dismiss? hide?). Either label it "Cancel" or remove it during active generation and only show it for queue items.

## History Bar

- [x] **Group thumbnails by prompt/session** — 20 images across 5 prompts becomes an undifferentiated wall. Add subtle dividers between sessions or show prompt on hover.
- [x] **Increase HISTORY label contrast** — Low-contrast label is easy to miss. Give it more visual weight if it's the entry point to full history.

## Layout & Theme

- [ ] **Consider collapsing sidebar sections** — Prompt, model, LoRA, dimensions, queue, GPU status is a lot for one panel. Consider moving LoRA + dimensions into a "Settings" expandable or second tab. Keep primary flow: prompt → model → generate.
- [ ] **Differentiate purple accent usage** — Purple currently means: selected state, generating state, button fill, active tab, queue pill. Introduce brightness/saturation variations or a secondary accent for hierarchy.
- [x] **Lazy-mount inactive tabs** — All tabs (Datasets, Models, Training, Outputs) are always mounted with running queries. Conditionally render heavy tabs; only keep GenerateView always-mounted.
