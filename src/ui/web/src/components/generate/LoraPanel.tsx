import { PlusIcon, XIcon } from 'lucide-react'
import { Button } from '@/components/ui/button'
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select'
import { Slider } from '@/components/ui/slider'
import type { InstalledModel } from '../../api'
import type { GenerateFormState, LoraEntry } from './generate-state'

type Props = {
  models: InstalledModel[]
  form: GenerateFormState
  setForm: React.Dispatch<React.SetStateAction<GenerateFormState>>
}

export function LoraPanel({ models, form, setForm }: Props) {
  const loras = models.filter((m) => m.model_type === 'lora')

  const addLora = () => {
    const first = loras[0]
    if (!first) return
    setForm((prev) => ({
      ...prev,
      loras: [...prev.loras, { id: first.id, name: first.name, strength: 0.8 }],
    }))
  }

  const updateLora = (idx: number, patch: Partial<LoraEntry>) => {
    setForm((prev) => ({
      ...prev,
      loras: prev.loras.map((entry, i) => (i === idx ? { ...entry, ...patch } : entry)),
    }))
  }

  const removeLora = (idx: number) => {
    setForm((prev) => ({
      ...prev,
      loras: prev.loras.filter((_, i) => i !== idx),
    }))
  }

  return (
    <div className="space-y-2">
      <div className="flex items-center justify-between">
        <span className="text-[11px] font-medium uppercase tracking-wider text-muted-foreground">
          LoRA
        </span>
        <Button
          type="button"
          variant="ghost"
          size="sm"
          className="h-6 gap-1 px-2 text-[11px]"
          onClick={addLora}
          disabled={loras.length === 0}
        >
          <PlusIcon className="size-3" />
          Add
        </Button>
      </div>

      {form.loras.length === 0 && (
        <p className="text-[11px] text-muted-foreground/60">
          {loras.length === 0 ? 'No LoRAs installed' : 'No LoRAs applied'}
        </p>
      )}

      <div className="space-y-2">
        {form.loras.map((entry, idx) => (
          <div
            key={`${entry.id}-${idx}`}
            className="group flex items-center gap-2 rounded-md border border-border/60 bg-secondary/20 px-2.5 py-1.5"
          >
            {/* LoRA selector */}
            <Select
              value={entry.id}
              onValueChange={(nextId) => {
                const model = loras.find((l) => l.id === nextId)
                updateLora(idx, { id: nextId, name: model?.name ?? nextId })
              }}
            >
              <SelectTrigger className="h-7 flex-1 border-0 bg-transparent px-1 text-xs shadow-none">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                {loras.map((model) => (
                  <SelectItem key={model.id} value={model.id}>
                    {model.name}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>

            {/* Strength slider */}
            <div className="flex w-36 items-center gap-2">
              <Slider
                value={[entry.strength]}
                min={0}
                max={2}
                step={0.05}
                className="flex-1"
                onValueChange={(next) => updateLora(idx, { strength: next[0] ?? entry.strength })}
              />
              <span className="w-9 text-right font-mono text-[10px] text-muted-foreground">
                {entry.strength.toFixed(2)}
              </span>
            </div>

            {/* Remove */}
            <Button
              type="button"
              variant="ghost"
              size="icon"
              className="size-6 opacity-0 transition-opacity group-hover:opacity-100"
              onClick={() => removeLora(idx)}
            >
              <XIcon className="size-3" />
            </Button>
          </div>
        ))}
      </div>
    </div>
  )
}
