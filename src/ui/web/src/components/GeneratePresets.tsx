import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import type { GenerateFormState } from './generate-form-state'

type Props = {
  form: GenerateFormState
  setForm: React.Dispatch<React.SetStateAction<GenerateFormState>>
}

type SizePreset = 'square' | 'portrait' | 'landscape' | 'custom'

export const SIZE_PRESETS: Record<Exclude<SizePreset, 'custom'>, { width: number; height: number }> = {
  square: { width: 1024, height: 1024 },
  portrait: { width: 832, height: 1216 },
  landscape: { width: 1216, height: 832 },
}

export function detectPreset(width: number, height: number): SizePreset {
  if (width === SIZE_PRESETS.square.width && height === SIZE_PRESETS.square.height) return 'square'
  if (width === SIZE_PRESETS.portrait.width && height === SIZE_PRESETS.portrait.height) return 'portrait'
  if (width === SIZE_PRESETS.landscape.width && height === SIZE_PRESETS.landscape.height) return 'landscape'
  return 'custom'
}

function parseNumber(value: string, fallback: number): number {
  const next = Number(value)
  return Number.isFinite(next) ? next : fallback
}

export function GeneratePresets({ form, setForm }: Props) {
  const sizePreset = detectPreset(form.width, form.height)

  const applyPreset = (preset: Exclude<SizePreset, 'custom'>) => {
    setForm((prev) => ({
      ...prev,
      width: SIZE_PRESETS[preset].width,
      height: SIZE_PRESETS[preset].height,
    }))
  }

  return (
    <aside className="self-start rounded-lg border border-border/80 bg-secondary/20 p-3 xl:sticky xl:top-20">
      <div className="space-y-2">
        <div className="text-xs uppercase tracking-wide text-muted-foreground">Size preset</div>
        <div className="grid grid-cols-2 gap-2">
          <Button
            type="button"
            size="sm"
            variant={sizePreset === 'square' ? 'secondary' : 'outline'}
            onClick={() => applyPreset('square')}
          >
            Square
          </Button>
          <Button
            type="button"
            size="sm"
            variant={sizePreset === 'portrait' ? 'secondary' : 'outline'}
            onClick={() => applyPreset('portrait')}
          >
            Portrait
          </Button>
          <Button
            type="button"
            size="sm"
            variant={sizePreset === 'landscape' ? 'secondary' : 'outline'}
            onClick={() => applyPreset('landscape')}
            className="col-span-2"
          >
            Landscape
          </Button>
        </div>
        {sizePreset === 'custom' ? (
          <div className="text-xs text-muted-foreground">Using custom dimensions</div>
        ) : null}
      </div>

      <div className="mt-4 grid gap-3">
        <label className="flex flex-col gap-2">
          <span className="text-xs uppercase tracking-wide text-muted-foreground">Width</span>
          <Input
            type="number"
            min={128}
            step={64}
            value={form.width}
            onChange={(e) =>
              setForm((prev) => ({ ...prev, width: parseNumber(e.target.value, prev.width) }))
            }
          />
        </label>

        <label className="flex flex-col gap-2">
          <span className="text-xs uppercase tracking-wide text-muted-foreground">Height</span>
          <Input
            type="number"
            min={128}
            step={64}
            value={form.height}
            onChange={(e) =>
              setForm((prev) => ({ ...prev, height: parseNumber(e.target.value, prev.height) }))
            }
          />
        </label>

        <label className="flex flex-col gap-2">
          <span className="text-xs uppercase tracking-wide text-muted-foreground">Images</span>
          <Input
            type="number"
            min={1}
            max={16}
            value={form.num_images}
            onChange={(e) =>
              setForm((prev) => ({ ...prev, num_images: parseNumber(e.target.value, prev.num_images) }))
            }
          />
        </label>
      </div>

      <GenerateAdvancedSettings form={form} setForm={setForm} />
    </aside>
  )
}

function GenerateAdvancedSettings({ form, setForm }: Props) {
  return (
    <details className="mt-4 rounded-md border border-border/70 px-3 py-2">
      <summary className="cursor-pointer text-sm font-medium">Advanced controls</summary>
      <div className="mt-3 space-y-3">
        <label className="flex flex-col gap-2">
          <span className="text-xs uppercase tracking-wide text-muted-foreground">
            Negative prompt
          </span>
          <textarea
            className="flex min-h-[80px] w-full rounded-md border border-input bg-background px-3 py-2 text-sm ring-offset-background placeholder:text-muted-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 disabled:cursor-not-allowed disabled:opacity-50"
            rows={3}
            value={form.negative_prompt}
            onChange={(e) => setForm((prev) => ({ ...prev, negative_prompt: e.target.value }))}
          />
        </label>

        <label className="flex flex-col gap-2">
          <span className="text-xs uppercase tracking-wide text-muted-foreground">Steps</span>
          <input
            className="flex h-10 w-full rounded-md border border-input bg-background px-3 py-2 text-sm ring-offset-background focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 disabled:cursor-not-allowed disabled:opacity-50"
            type="number"
            min={1}
            max={200}
            value={form.steps}
            onChange={(e) =>
              setForm((prev) => ({ ...prev, steps: parseNumber(e.target.value, prev.steps) }))
            }
          />
        </label>

        <label className="flex flex-col gap-2">
          <span className="text-xs uppercase tracking-wide text-muted-foreground">Guidance</span>
          <input
            className="flex h-10 w-full rounded-md border border-input bg-background px-3 py-2 text-sm ring-offset-background focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 disabled:cursor-not-allowed disabled:opacity-50"
            type="number"
            min={1}
            max={30}
            step={0.1}
            value={form.guidance}
            onChange={(e) =>
              setForm((prev) => ({ ...prev, guidance: parseNumber(e.target.value, prev.guidance) }))
            }
          />
        </label>

        <label className="flex flex-col gap-2">
          <span className="text-xs uppercase tracking-wide text-muted-foreground">Seed</span>
          <div className="flex gap-2">
            <input
              className="flex h-10 w-full rounded-md border border-input bg-background px-3 py-2 text-sm ring-offset-background focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 disabled:cursor-not-allowed disabled:opacity-50"
              type="number"
              min={0}
              value={form.seed}
              onChange={(e) =>
                setForm((prev) => ({ ...prev, seed: parseNumber(e.target.value, prev.seed) }))
              }
            />
            <Button
              type="button"
              variant="outline"
              onClick={() =>
                setForm((prev) => ({
                  ...prev,
                  seed: Math.floor(Math.random() * 4_294_967_295),
                }))
              }
            >
              Random
            </Button>
          </div>
        </label>
      </div>
    </details>
  )
}
