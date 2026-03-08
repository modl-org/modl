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

/** Extract step number from LoRA names like "art-style-v4_000004070" → 4070 */
function parseStep(name: string): number | null {
  const m = name.match(/_0*(\d+)$/)
  return m ? parseInt(m[1], 10) : null
}

/** Format a name for display — strip leading zeros from step suffix */
function displayName(name: string): { base: string; step: number | null } {
  const step = parseStep(name)
  if (step != null) {
    const base = name.replace(/_\d+$/, '')
    return { base, step }
  }
  return { base: name, step: null }
}

export function LoraPanel({ models, form, setForm }: Props) {
  const loras = models.filter((m) => m.model_type === 'lora')

  /** Insert a trigger word into the prompt (at the start, if not already present) */
  const insertTriggerWord = (word: string) => {
    setForm((prev) => {
      const trimmed = prev.prompt.trim()
      if (trimmed.toLowerCase().includes(word.toLowerCase())) return prev
      const newPrompt = trimmed ? `${word} ${trimmed}` : word
      return { ...prev, prompt: newPrompt }
    })
  }

  const addLora = () => {
    const first = loras[0]
    if (!first) return
    setForm((prev) => ({
      ...prev,
      loras: [...prev.loras, { id: first.id, name: first.name, strength: 0.8 }],
    }))
    if (first.trigger_word) insertTriggerWord(first.trigger_word)
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
        <span className="text-[11px] font-medium text-muted-foreground">LoRA</span>
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
        {form.loras.map((entry, idx) => {
          const loraModel = loras.find((l) => l.id === entry.id)
          const triggerWord = loraModel?.trigger_word
          const sampleUrl = loraModel?.sample_image_url
          const { base, step } = displayName(entry.name)

          return (
            <div
              key={`${entry.id}-${idx}`}
              className="group rounded-md border border-border/60 bg-secondary/20 p-2"
            >
              {/* Top row: selector + remove */}
              <div className="flex items-center gap-2">
                <Select
                  value={entry.id}
                  onValueChange={(nextId) => {
                    const model = loras.find((l) => l.id === nextId)
                    updateLora(idx, { id: nextId, name: model?.name ?? nextId })
                    if (model?.trigger_word) insertTriggerWord(model.trigger_word)
                  }}
                >
                  <SelectTrigger className="h-7 flex-1 border-0 bg-transparent px-1 text-xs shadow-none">
                    <SelectValue>
                      <span className="flex items-center gap-2 truncate">
                        {sampleUrl && (
                          <img
                            src={`/files/${sampleUrl}`}
                            alt=""
                            className="size-5 flex-shrink-0 rounded object-cover"
                          />
                        )}
                        <span className="truncate">{base}</span>
                        {step != null && (
                          <span className="flex-shrink-0 rounded bg-secondary/60 px-1 py-0.5 font-mono text-[9px] text-muted-foreground">
                            step {step.toLocaleString()}
                          </span>
                        )}
                      </span>
                    </SelectValue>
                  </SelectTrigger>
                  <SelectContent>
                    {loras.map((model) => {
                      const d = displayName(model.name)
                      return (
                        <SelectItem key={model.id} value={model.id}>
                          <span className="flex items-center gap-2">
                            {model.sample_image_url && (
                              <img
                                src={`/files/${model.sample_image_url}`}
                                alt=""
                                className="size-5 rounded object-cover"
                              />
                            )}
                            <span className="flex flex-col">
                              <span className="flex items-center gap-1.5">
                                <span>{d.base}</span>
                                {d.step != null && (
                                  <span className="rounded bg-secondary/60 px-1 py-0.5 font-mono text-[9px] text-muted-foreground">
                                    {d.step.toLocaleString()}
                                  </span>
                                )}
                              </span>
                              {model.trigger_word && (
                                <span className="font-mono text-[9px] text-muted-foreground/60">
                                  {model.trigger_word}
                                </span>
                              )}
                            </span>
                          </span>
                        </SelectItem>
                      )
                    })}
                  </SelectContent>
                </Select>

                <Button
                  type="button"
                  variant="ghost"
                  size="icon"
                  className="size-6 flex-shrink-0 opacity-0 transition-opacity group-hover:opacity-100"
                  onClick={() => removeLora(idx)}
                >
                  <XIcon className="size-3" />
                </Button>
              </div>

              {/* Strength slider */}
              <div className="mt-1.5 flex items-center gap-2 pl-0.5">
                <Slider
                  value={[entry.strength]}
                  min={0}
                  max={2}
                  step={0.05}
                  className="flex-1"
                  onValueChange={(next) =>
                    updateLora(idx, { strength: next[0] ?? entry.strength })
                  }
                />
                <span className="w-9 text-right font-mono text-[10px] text-muted-foreground">
                  {entry.strength.toFixed(2)}
                </span>
              </div>

              {/* Trigger word chip */}
              {triggerWord && (
                <div className="mt-1.5 flex items-center gap-1.5">
                  <span className="text-[10px] text-muted-foreground/60">trigger:</span>
                  <button
                    type="button"
                    onClick={() => insertTriggerWord(triggerWord)}
                    className="rounded bg-primary/10 px-1.5 py-0.5 font-mono text-[10px] font-medium text-primary transition-colors hover:bg-primary/20"
                    title="Click to insert into prompt"
                  >
                    {triggerWord}
                  </button>
                </div>
              )}
            </div>
          )
        })}
      </div>
    </div>
  )
}
