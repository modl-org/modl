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
import type { InstalledModel, ModelFamily } from '../../api'
import { findModelFamily, type GenerateFormState, type LoraEntry } from './generate-state'

type Props = {
  models: InstalledModel[]
  families: ModelFamily[]
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

/** Find the parent ModelFamily for a ModelFamilyInfo */
function findParentFamily(
  info: import('../../api').ModelFamilyInfo,
  families: ModelFamily[],
): ModelFamily | undefined {
  return families.find((f) => f.models.some((m) => m.id === info.id))
}

/** Check if a LoRA is compatible with the selected base model.
 *  LoRAs trained on the same family (e.g. flux-dev LoRA on flux-schnell) are compatible. */
function isLoraCompatible(
  lora: InstalledModel,
  selectedModelName: string,
  families: ModelFamily[],
): boolean {
  const selectedFamily = findModelFamily(selectedModelName, families)
  if (!selectedFamily) return true // can't determine selected model = show it

  const selectedParent = findParentFamily(selectedFamily, families)
  if (!selectedParent) return true

  // Try base_model_id first (set by artifact metadata for trained LoRAs)
  if (lora.base_model_id) {
    const loraFamily = findModelFamily(lora.base_model_id, families)
    if (loraFamily) {
      const loraParent = findParentFamily(loraFamily, families)
      return loraParent?.id === selectedParent.id
    }
  }

  // Fall back to depends_on (set by registry for pulled LoRAs)
  if (lora.depends_on && lora.depends_on.length > 0) {
    // Check if any dependency is in the same family as the selected model
    for (const dep of lora.depends_on) {
      const depFamily = findModelFamily(dep.id, families)
      if (depFamily) {
        const depParent = findParentFamily(depFamily, families)
        if (depParent) {
          return depParent.id === selectedParent.id
        }
      }
    }
    // Has dependencies but none matched a family — likely incompatible
    return false
  }

  // Also try matching by LoRA name itself (e.g. "qwen-image-lightning_step4070")
  const loraFamily = findModelFamily(lora.name, families)
  if (loraFamily) {
    const loraParent = findParentFamily(loraFamily, families)
    if (loraParent) {
      return loraParent.id === selectedParent.id
    }
  }

  return true // truly unknown = show it
}

/** Mini toggle switch */
function Toggle({ checked, onChange }: { checked: boolean; onChange: (v: boolean) => void }) {
  return (
    <button
      type="button"
      onClick={() => onChange(!checked)}
      className={`relative h-4 w-7 shrink-0 rounded-full transition-colors ${
        checked ? 'bg-primary' : 'bg-muted-foreground/20'
      }`}
      title={checked ? 'Enabled — click to disable' : 'Disabled — click to enable'}
    >
      <span
        className={`absolute top-0.5 h-3 w-3 rounded-full bg-white shadow-sm transition-[left] ${
          checked ? 'left-[13px]' : 'left-0.5'
        }`}
      />
    </button>
  )
}

export function LoraPanel({ models, families, form, setForm }: Props) {
  const allLoras = models.filter((m) => m.model_type === 'lora')
  const selectedModel = models.find((m) => m.id === form.base_model_id)

  // Split into compatible and incompatible
  const compatibleLoras = allLoras.filter((l) =>
    isLoraCompatible(l, selectedModel?.name ?? '', families),
  )
  const loras = compatibleLoras

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
      loras: [...prev.loras, { id: first.id, name: first.name, strength: 0.8, enabled: true }],
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

      {form.loras.length === 0 && allLoras.length === 0 && (
        <p className="text-[11px] text-muted-foreground/60">No LoRAs installed</p>
      )}
      {form.loras.length === 0 && allLoras.length > 0 && loras.length === 0 && (
        <p className="text-[11px] text-muted-foreground/60">
          No compatible LoRAs for this model
        </p>
      )}

      <div className="space-y-2">
        {form.loras.map((entry, idx) => {
          const loraModel = allLoras.find((l) => l.id === entry.id)
          const triggerWord = loraModel?.trigger_word
          const sampleUrl = loraModel?.sample_image_url
          const { base, step } = displayName(entry.name)
          const isEnabled = entry.enabled !== false

          return (
            <div
              key={`${entry.id}-${idx}`}
              className={`group rounded-md border p-2 transition-opacity ${
                isEnabled
                  ? 'border-border/60 bg-secondary/20'
                  : 'border-border/30 bg-secondary/5 opacity-50'
              }`}
            >
              {/* Top row: toggle + selector + remove */}
              <div className="flex items-center gap-2">
                <Toggle
                  checked={isEnabled}
                  onChange={(v) => updateLora(idx, { enabled: v })}
                />

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
                    {allLoras.map((model) => {
                      const d = displayName(model.name)
                      const compatible = isLoraCompatible(model, selectedModel?.name ?? '', families)
                      return (
                        <SelectItem key={model.id} value={model.id} disabled={!compatible}>
                          <span className="flex items-center gap-2">
                            {model.sample_image_url && (
                              <img
                                src={`/files/${model.sample_image_url}`}
                                alt=""
                                className={`size-5 rounded object-cover ${!compatible ? 'opacity-40 grayscale' : ''}`}
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
                                {!compatible && (
                                  <span className="rounded bg-destructive/10 px-1 py-0.5 text-[8px] text-destructive/60">
                                    incompatible
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

              {/* Strength slider — only interactive when enabled */}
              <div className={`mt-1.5 flex items-center gap-2 pl-0.5 ${!isEnabled ? 'pointer-events-none' : ''}`}>
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
