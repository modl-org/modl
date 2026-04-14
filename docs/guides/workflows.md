# Workflows — `modl run`

Batch image generation from a single YAML file. Declare a list of generate/edit
steps, hit run, walk away, come back to finished artifacts. Think "Makefile for
image generation."

> **Note for maintainers:** This guide is written to be extractable for the
> public modl-site marketing docs after the feature is polished and shipped.
> The "Why workflows", "Quickstart", and "Real example" sections are designed
> to drop cleanly into an MDX page. Keep them self-contained and narrative.

---

## Why workflows

If you're generating one image at a time, `modl generate "a cat"` is the right
tool. If you're producing a **collection** — a book chapter, a product sheet,
a mood board, a series of variations you'll pick winners from — you quickly hit
the "I'm retyping the same flags with different prompts" problem.

A workflow is a YAML file that declares the whole batch up front:

- **Model + LoRA once, used by every step.** No retyping.
- **Seed exploration.** Generate N variations of a prompt with explicit seeds so
  you can pick the winner later and reproduce it.
- **Cross-step references.** Step 3 can edit the output of step 1 without
  juggling file paths.
- **Shared defaults** (size, steps, guidance) with per-step overrides.
- **One run, one batch.** Kick it off, walk away. Everything lands in
  `~/.modl/outputs/<date>/` and shows up in `modl serve` alongside your regular
  generations.

It's the same batch pipeline you'd script yourself, but declarative, portable
across local and cloud (cloud path coming soon), and reproducible by seed.

## Quickstart

Create a file `hello-workflow.yaml`:

```yaml
name: hello
model: sdxl

defaults:
  width: 1024
  height: 1024
  steps: 28
  guidance: 7.0

steps:
  - id: cat
    generate: "a tiny cat reading a book"
    seed: 42

  - id: dog
    generate: "a tiny dog catching a frisbee"
    seed: 99
```

Then:

```bash
modl run hello-workflow.yaml
```

You'll see each step execute in order with live progress, and the two PNGs
will land in `~/.modl/outputs/<today>/` where `modl serve` can see them.

## The spec, field by field

### Top-level

```yaml
name: <string>           # required — used for display and DB labels
model: <model-id>        # required — e.g. "flux2-klein-4b", "sdxl", "qwen-image"
lora: <lora-id>          # optional — applied to every step
defaults:                # optional — inherited by every step unless overridden
  seed: <int>
  width: <int>
  height: <int>
  steps: <int>
  guidance: <float>
  count: <int>
steps:                   # required — ordered list, one or more
  - ...
```

### Steps

Every step has a unique `id` and is either a **generate** or an **edit** step
(exactly one — never both). Edit steps also require a `prompt:` field.

**Generate step:**

```yaml
- id: scene-1                          # required — must be unique within the workflow
  generate: "a misty forest at dawn"   # required — the prompt
  seed: 42                             # optional — overrides defaults.seed
  seeds: [42, 7, 99, 333]              # optional — mutually exclusive with seed+count
  width: 1024                          # optional — overrides defaults.width
  height: 1024                         # optional — overrides defaults.height
  steps: 4                             # optional — overrides defaults.steps
  guidance: 1.0                        # optional — overrides defaults.guidance
  count: 2                             # optional — overrides defaults.count
```

**Edit step:**

```yaml
- id: rain-variant
  edit: "./draft.png"                  # required — local path OR $step-id.outputs[N]
  prompt: "add heavy rain"             # required — the edit instruction
  seed: 42                             # same optional fields as generate
  seeds: [1, 2, 3]                     # ...including seeds
```

### Defaults and overrides

Anything declared in `defaults:` becomes the step value when the step doesn't
set it. Any field set on the step wins over the default.

```yaml
defaults:
  seed: 42
  steps: 28
steps:
  - id: a
    generate: "foo"                    # uses defaults: seed=42, steps=28
  - id: b
    generate: "bar"
    seed: 999                          # overrides: seed=999, steps=28 (from defaults)
```

If neither defaults nor the step sets `steps`/`guidance`/`width`/`height`,
modl falls back to the model's sensible defaults from `models.toml` (e.g.
Klein 4B → 4 steps, guidance 1.0, 1024×1024).

### Seeds: exploration vs noise variation

Two different use cases, two different fields:

**`seeds: [42, 7, 99, 333]`** — explicit variation exploration. Runs one
generation per seed. Every output is reproducible by its seed number. This is
what you want when making a book chapter and you'll pick winners later.

**`seed: 42` + `count: 4`** — four images at the same seed. The worker varies
internal noise per image, so you get a small spread without changing the base
seed. Useful when you have a prompt you trust and just want a few options.

Setting **both** `seed` and `seeds` is a parse error — pick one. Setting
`seeds` with `count` is also a parse error, since `seeds.len()` is the count.

### Cross-step references: `$step-id.outputs[N]`

Edit steps can consume earlier step outputs without knowing the file path:

```yaml
steps:
  - id: forest
    generate: "a misty cabin in a forest"
    seeds: [42, 7, 99]           # produces 3 outputs

  - id: rain-variant
    edit: "$forest.outputs[0]"   # refers to the first forest output (seed=42)
    prompt: "add heavy rain"

  - id: lightning-variant
    edit: "$forest.outputs[2]"   # refers to the third forest output (seed=99)
    prompt: "add lightning in the background"
```

**Rules:**

- References must point to **earlier** steps (no forward references, no self-references).
- The `[N]` index is resolved at runtime. `[0]` is always the first output
  from the referenced step; if `seeds: [...]` was used, the index follows the
  seed order you declared.
- If `[N]` is out of bounds (e.g. the referenced step produced fewer outputs
  than expected), you get a clean runtime error — the workflow fails at that
  step, nothing downstream runs.
- Bounds can't be checked at parse time because `count` / `seeds.len()`
  determines cardinality at execution time.

## Where outputs go

Every artifact lands in `~/.modl/outputs/<YYYY-MM-DD>/`, flat. No per-workflow
subdirectory. **This is deliberate.**

- `modl serve` already scans the date directory — your workflow outputs appear
  in the Outputs tab alongside anything you generated manually with
  `modl generate`.
- Each step becomes a row in the local jobs database with labels
  `workflow=<name>`, `workflow_run=<timestamp-name>`, `workflow_step=<id>`.
- `modl serve` gets full spec + prompt + seed for every image, same as regular
  generations.

If you need to find just one workflow's outputs, query the DB directly:

```bash
sqlite3 ~/.modl/state.db \
  "SELECT spec_json FROM jobs WHERE json_extract(spec_json, '\$.labels.workflow') = 'book-chapter-3'"
```

A dedicated "workflow runs" view in `modl serve` is on the roadmap — the data
is already there, just needs a UI.

## LoRA support

Declare a LoRA at the top level and it applies to every step:

```yaml
name: character-sheet
model: qwen-image
lora: qwen-image-2512-lightning

defaults:
  steps: 8
  guidance: 1.0

steps:
  - id: front
    generate: "character portrait, front view, white background"
  - id: side
    generate: "character portrait, side profile, white background"
  - id: back
    generate: "character portrait, back view, white background"
```

The LoRA is resolved **once** at the start of the run. If it's not installed
locally, the workflow fails immediately before any compute happens — you fix
the problem and re-run, no wasted GPU time.

Per-step LoRA override (different LoRAs for different steps) is not currently
supported. If you need it, open an issue with the use case.

## Real example: KDP book chapter

This is the actual use case the feature was built for — generating scenes for
a print-on-demand children's book where you want seed variations so you can
pick the winner for each spread.

```yaml
# book-chapter-3.yaml
name: book-chapter-3-treehouse
model: flux2-klein-4b
lora: my-son-v2               # trained character LoRA

defaults:
  width: 1024
  height: 1024
  steps: 4
  guidance: 1.0

steps:
  # Page 1: hero climbs the treehouse ladder
  - id: climb
    generate: "OHWX climbing a wooden ladder to a treehouse, morning light, children's book illustration style"
    seeds: [10, 20, 30, 40]     # 4 variations, pick winner later

  # Page 2: interior reveal
  - id: interior
    generate: "inside a cozy wooden treehouse, OHWX peeking through a window, warm afternoon light, children's book illustration style"
    seeds: [11, 21, 31, 41]

  # Page 3: sunset from the treehouse window
  - id: sunset
    generate: "view from a treehouse window at sunset, golden hour, wistful mood, children's book illustration style"
    seeds: [12, 22, 32, 42]

  # Rainy-day variation of page 1 (references the first winner)
  - id: climb-rainy
    edit: "$climb.outputs[0]"
    prompt: "add gentle rain and wet leaves, still children's book illustration style"
    seed: 10
```

Run it:

```bash
modl run book-chapter-3.yaml
```

Output: 13 artifacts (4+4+4+1) in `~/.modl/outputs/<today>/`. Open
`modl serve`, flip to the Outputs tab, pick your winner per scene. If you want
to re-run a single scene with a new seed, you know the prompts and seeds —
they're in the spec file.

The edit step (`climb-rainy`) uses `$climb.outputs[0]` to reference the
*first* variation of `climb` (seed=10). If you later decide seed=20 was the
winner, change the index to `[1]` and re-run — no path juggling.

## Troubleshooting

### "LoRA `my-son-v2` not found in local store"

The LoRA isn't pulled. Train it (`modl train ...`) or pull it if it's on the
hub (`modl pull hub:username/my-son-v2`). The error happens at workflow start
before any GPU work, so you don't waste compute.

### "Model `flux-dev` is not installed locally"

Same idea — run `modl pull flux-dev` first.

### "step `forest`: local image `./input.png` does not exist"

Edit step source path is relative to the workflow YAML file's directory, not
your current working directory. Either use an absolute path or put `input.png`
next to the YAML file.

### "step `forest` produced 3 output(s), index [5] is out of bounds"

You referenced `$forest.outputs[5]` but the forest step only produced 3
outputs. Check your `seeds` or `count` on the referenced step.

### "duplicate step id `scene-1` at step 3"

Two steps share the same id. Ids must be unique within a workflow — renaming
fixes it.

### Images don't appear in `modl serve`

- Are they in today's date directory? `ls ~/.modl/outputs/$(date +%Y-%m-%d)/`
- Is the UI pointed at the right data dir? Check `modl serve` startup logs for
  the outputs path.
- Did the worker actually emit `Artifact` events? Check the workflow run's
  stderr — if a step says `✓ 0 artifacts`, the worker failed silently.

### Workflow stops mid-run after one step fails

That's the design — **fail fast, don't continue on broken state**. Fix the
error, re-run the full workflow. Step resumption is not supported today.

## What's not in workflows (yet)

These are out of scope for the current implementation. Several are planned;
open an issue before assuming something is coming.

- **`modl run --cloud`** — run the whole workflow on a rented GPU instead of
  locally. Architected, coming in the next phase. See
  `docs/plans/workflow-run.md`.
- **Upscale / analysis steps** — `upscale`, `face-restore`, `score`, etc. as
  first-class step kinds. Deferred to Phase B.2.
- **Per-step LoRA or model override** — one workflow uses one model + one
  LoRA for all steps. If this becomes a real blocker, we'll add it.
- **Conditionals, variables, templating** — no `if:`, no `{{ }}`. If the spec
  can't be a flat ordered list, it's probably two workflows.
- **Parallel step execution** — sequential only.
- **Step resumption after failure** — re-run the whole workflow.
- **`modl run --watch`** — no file-watcher mode.
- **Workflow-grouped view in `modl serve`** — labels are in the DB for when
  the UI catches up.

## Reference

- **Plan doc:** `docs/plans/workflow-run.md` (private)
- **Smoke tests:** `docs/plans/smoke-test-workflow.md` (private)
- **Source:**
  - Parser / validator: `src/core/workflow.rs`
  - CLI + execution: `src/cli/run.rs`
- **Related commands:** `modl generate`, `modl edit`, `modl serve`

## Feedback

If you build a workflow that hits a wall — a missing field, awkward syntax,
unclear error — say so. The spec is deliberately minimal; the only way to know
what's missing is to use it on real work.
