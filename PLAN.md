# PLAN.md — Future Ideas & Roadmap

This document captures planned improvements and feature ideas that go beyond
the current implementation. Items here are aspirational — they may or may not
be built, and the design may change.

---

## LoRA Management & Generation UX

### Implemented

- **Trigger word display** — Trained LoRAs show their trigger word in the
  generate form. Clicking the chip inserts it into the prompt.
- **Sample thumbnails** — Trained LoRAs show a thumbnail from their latest
  training sample in the selector and in the active LoRA card.
- **Base model metadata** — The `/api/models` endpoint returns `trigger_word`,
  `base_model_id`, and `sample_image_url` for LoRAs that have artifact records.

### Future Ideas

- **Intermediate step management** — Training produces many intermediate
  checkpoints (e.g. `art-style-v4_000002035`, `_000004070`, etc.) that get
  installed as separate LoRAs. These clutter the LoRA list. Ideas:
  - Group intermediates under their parent training run in the UI
  - Let users "promote" a specific step to be the primary LoRA
  - Auto-hide intermediates, show only the final or promoted step
  - `modl prune-intermediates <run>` CLI command to remove all but the best

- **LoRA comparison view** — Side-by-side generation with different LoRAs or
  different steps of the same training run. Useful for picking the best
  checkpoint. Could show a grid: rows = steps, columns = different prompts.

- **External LoRA import with metadata** — When importing a LoRA from
  CivitAI or HuggingFace, scrape/store trigger words and sample images.
  The `modl pull` flow could extract this from the registry manifest or
  from model card metadata.

- **Favorite LoRAs** — Pin frequently-used LoRAs to the top of the selector.
  Could reuse the existing `favorites` table in state.db.

- **LoRA compatibility warnings** — Warn when a LoRA's `base_model_id` doesn't
  match the currently selected checkpoint (e.g. SDXL LoRA with SD1.5 base).

- **Trigger word auto-insert** — When a LoRA is added to the form, automatically
  prepend its trigger word to the prompt (with an undo toast). Currently the user
  must click the chip manually.

- **LoRA strength presets** — Save per-LoRA preferred strength values so they
  persist across sessions. Currently defaults to 0.8 every time.

- **Batch LoRA testing** — Generate the same prompt with varying LoRA strengths
  (0.4, 0.6, 0.8, 1.0) in one batch to find the sweet spot.

---

## Generation

- **Flux text encoder fix** — `from_single_file()` doesn't load text encoders
  for Flux models. Needs explicit text encoder loading in the Python executor.
  SDXL works fine; Flux is blocked on this.

- **Generation queue** — Allow queueing multiple generation requests instead of
  blocking on one at a time. The current `AtomicBool` lock prevents concurrent
  runs; a proper job queue would let users stack requests.

- **Generation history search** — Filter/search past generations by prompt,
  model, LoRA, or date. Currently just shows a flat chronological gallery.

- **A/B prompt comparison** — Generate two prompts side-by-side with identical
  settings to compare prompt phrasing.

---

## Training

- **Live loss chart** — The training status already streams loss values. Render
  a sparkline or mini chart in the UI instead of just the current number.

- **Training presets** — Save and load training configurations (learning rate,
  steps, resolution, etc.) as named presets.

- **Resume from UI** — Currently resuming a training run requires the CLI.
  Add a "Resume" button in the training detail view.

---

## Infrastructure

- **Daemonized training** — Run training as a background daemon instead of a
  child process, so it survives terminal disconnects without tmux/screen.

- **Multi-GPU support** — Detect and offer GPU selection when multiple GPUs
  are available.

- **Remote generation** — Send generation requests to a remote GPU (Modal,
  RunPod) when local GPU is unavailable or too slow.
