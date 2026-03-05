import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { SIZE_PRESETS, detectSizePreset, type GenerateFormState } from './generate-state'

type Props = {
  form: GenerateFormState
  setForm: React.Dispatch<React.SetStateAction<GenerateFormState>>
}

function parseNum(val: string, fallback: number): number {
  const n = Number(val)
  return Number.isFinite(n) && n > 0 ? n : fallback
}

export function SizePanel({ form, setForm }: Props) {
  const activePreset = detectSizePreset(form.width, form.height)

  const applyPreset = (width: number, height: number) => {
    setForm((prev) => ({ ...prev, width, height }))
  }

  return (
    <div className="space-y-2.5">
      <span className="text-[11px] font-medium uppercase tracking-wider text-muted-foreground">
        Size
      </span>

      {/* Preset buttons */}
      <div className="flex gap-1">
        {SIZE_PRESETS.map((preset) => (
          <Button
            key={preset.label}
            type="button"
            size="sm"
            variant={activePreset === preset.label ? 'secondary' : 'outline'}
            className="h-7 flex-1 px-1 text-[11px]"
            onClick={() => applyPreset(preset.width, preset.height)}
          >
            {preset.label}
          </Button>
        ))}
      </div>

      {/* Custom dimensions */}
      <div className="grid grid-cols-2 gap-2">
        <label className="flex flex-col gap-1">
          <span className="text-[10px] text-muted-foreground/70">W</span>
          <Input
            type="number"
            min={128}
            max={2048}
            step={64}
            value={form.width}
            className="h-7 bg-background/60 text-xs"
            onChange={(e) =>
              setForm((prev) => ({ ...prev, width: parseNum(e.target.value, prev.width) }))
            }
          />
        </label>
        <label className="flex flex-col gap-1">
          <span className="text-[10px] text-muted-foreground/70">H</span>
          <Input
            type="number"
            min={128}
            max={2048}
            step={64}
            value={form.height}
            className="h-7 bg-background/60 text-xs"
            onChange={(e) =>
              setForm((prev) => ({ ...prev, height: parseNum(e.target.value, prev.height) }))
            }
          />
        </label>
      </div>

      {activePreset === 'custom' && (
        <p className="text-[10px] text-muted-foreground/50">Custom dimensions</p>
      )}
    </div>
  )
}
