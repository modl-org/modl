import { DicesIcon } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Slider } from '@/components/ui/slider'
import { randomSeed, type GenerateFormState } from './generate-state'

type Props = {
  form: GenerateFormState
  setForm: React.Dispatch<React.SetStateAction<GenerateFormState>>
}

function parseNum(val: string, fallback: number): number {
  const n = Number(val)
  return Number.isFinite(n) ? n : fallback
}

export function SamplingPanel({ form, setForm }: Props) {
  return (
    <div className="space-y-3">
      {/* Section labels removed — handled by CollapsibleSection */}

      {/* Steps */}
      <div className="space-y-1">
        <div className="flex items-center justify-between">
          <span className="text-[11px] font-medium text-muted-foreground">Steps</span>
          <span className="font-mono text-[11px] text-foreground/70">{form.steps}</span>
        </div>
        <Slider
          value={[form.steps]}
          min={1}
          max={150}
          step={1}
          onValueChange={([v]) => setForm((prev) => ({ ...prev, steps: v ?? prev.steps }))}
        />
      </div>

      {/* Guidance / CFG Scale */}
      <div className="space-y-1">
        <div className="flex items-center justify-between">
          <span className="text-[11px] font-medium text-muted-foreground">CFG Scale</span>
          <span className="font-mono text-[11px] text-foreground/70">
            {form.guidance.toFixed(1)}
          </span>
        </div>
        <Slider
          value={[form.guidance]}
          min={0}
          max={30}
          step={0.5}
          onValueChange={([v]) => setForm((prev) => ({ ...prev, guidance: v ?? prev.guidance }))}
        />
      </div>

      {/* Seed */}
      <div className="space-y-1">
        <span className="text-[11px] font-medium text-muted-foreground">Seed</span>
        <div className="flex gap-1.5">
          <Input
            type="number"
            min={0}
            value={form.seed}
            className="h-7 flex-1 bg-background/60 font-mono text-xs"
            onChange={(e) =>
              setForm((prev) => ({ ...prev, seed: parseNum(e.target.value, prev.seed) }))
            }
          />
          <Button
            type="button"
            variant="outline"
            size="icon"
            className="size-7 shrink-0"
            onClick={() => setForm((prev) => ({ ...prev, seed: randomSeed() }))}
            title="Randomize seed"
          >
            <DicesIcon className="size-3.5" />
          </Button>
        </div>
      </div>
    </div>
  )
}
