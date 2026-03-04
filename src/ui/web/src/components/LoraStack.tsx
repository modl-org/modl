import { Button } from '@/components/ui/button'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select'
import { Slider } from '@/components/ui/slider'
import type { InstalledModel } from '../api'

export type LoraEntry = {
  id: string
  strength: number
}

type Props = {
  loras: InstalledModel[]
  value: LoraEntry[]
  onChange: (entries: LoraEntry[]) => void
}

export function LoraStack({ loras, value, onChange }: Props) {
  const add = () => {
    const first = loras[0]?.id
    if (!first) return
    onChange([...value, { id: first, strength: 0.8 }])
  }

  const update = (idx: number, patch: Partial<LoraEntry>) => {
    const next = value.map((entry, i) => (i === idx ? { ...entry, ...patch } : entry))
    onChange(next)
  }

  const remove = (idx: number) => {
    onChange(value.filter((_, i) => i !== idx))
  }

  return (
    <div className="space-y-3">
      <div className="flex items-center justify-between">
        <span className="text-xs uppercase tracking-wide text-muted-foreground">LoRA stack</span>
        <Button type="button" variant="outline" size="sm" onClick={add} disabled={loras.length === 0}>
          Add LoRA
        </Button>
      </div>

      {value.length === 0 ? (
        <p className="text-sm text-muted-foreground">No LoRA modifiers selected.</p>
      ) : null}

      <div className="space-y-3">
        {value.map((entry, idx) => (
          <div
            key={`${entry.id}-${idx}`}
            className="grid gap-3 rounded-lg border border-border/80 bg-secondary/30 p-3 md:grid-cols-[1fr_220px_70px_auto] md:items-center"
          >
            <Select value={entry.id} onValueChange={(nextId) => update(idx, { id: nextId })}>
              <SelectTrigger className="w-full bg-background">
                <SelectValue placeholder="Select LoRA" />
              </SelectTrigger>
              <SelectContent>
                {loras.map((model) => (
                  <SelectItem key={model.id} value={model.id}>
                    {model.name}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>

            <Slider
              value={[entry.strength]}
              min={0}
              max={2}
              step={0.05}
              onValueChange={(next) => update(idx, { strength: next[0] ?? entry.strength })}
            />

            <span className="font-mono text-xs text-muted-foreground">{entry.strength.toFixed(2)}</span>

            <Button type="button" variant="ghost" size="sm" onClick={() => remove(idx)}>
              Remove
            </Button>
          </div>
        ))}
      </div>
    </div>
  )
}
