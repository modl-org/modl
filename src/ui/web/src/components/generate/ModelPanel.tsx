import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select'
import type { InstalledModel, ModelFamily } from '../../api'
import { modelDefaults, findModelFamily, type GenerateFormState } from './generate-state'

type Props = {
  models: InstalledModel[]
  families: ModelFamily[]
  form: GenerateFormState
  setForm: React.Dispatch<React.SetStateAction<GenerateFormState>>
  /** When true, auto-adjust steps/guidance on model change */
  autoDefaults?: boolean
}

/** Quality/speed dots (1-5) */
function RatingDots({ value, max = 5 }: { value: number; max?: number }) {
  return (
    <span className="inline-flex gap-0.5">
      {Array.from({ length: max }, (_, i) => (
        <span
          key={i}
          className={`inline-block size-1.5 rounded-full ${
            i < value ? 'bg-primary/80' : 'bg-muted-foreground/20'
          }`}
        />
      ))}
    </span>
  )
}

export function ModelPanel({ models, families, form, setForm, autoDefaults = true }: Props) {
  const checkpoints = models.filter((m) => m.model_type === 'checkpoint' || m.model_type === 'diffusion_model')

  const handleChange = (modelId: string) => {
    const model = checkpoints.find((m) => m.id === modelId)
    if (!model) return

    setForm((prev) => {
      const patch: Partial<GenerateFormState> = { base_model_id: modelId }

      if (autoDefaults) {
        const info = findModelFamily(model.name, families)
        const defaults = modelDefaults(model.name, info)
        patch.steps = defaults.steps
        patch.guidance = defaults.guidance
      }

      return { ...prev, ...patch }
    })
  }

  return (
    <div className="space-y-1.5">
      <Select
        value={form.base_model_id}
        onValueChange={handleChange}
        disabled={checkpoints.length === 0}
      >
        <SelectTrigger className="w-full bg-background/60">
          <SelectValue placeholder={checkpoints.length === 0 ? 'No checkpoints installed' : 'Select a model'} />
        </SelectTrigger>
        <SelectContent>
          {checkpoints.map((model) => {
            const info = findModelFamily(model.name, families)
            const sizeGB = (model.size_bytes / 1024 / 1024 / 1024).toFixed(1)
            const triggerLabel = [
              model.name,
              model.variant,
              info ? `${info.total_b}B` : null,
            ].filter(Boolean).join(' · ')
            return (
              <SelectItem key={model.id} value={model.id} textValue={triggerLabel} className="py-2">
                <div className="flex flex-col gap-0.5">
                  {/* Line 1: model name + param count */}
                  <div className="flex items-center gap-2">
                    <span className="font-medium">{model.name}</span>
                    {info && (
                      <span className="text-[10px] text-muted-foreground/60">
                        {info.total_b}B
                      </span>
                    )}
                  </div>
                  {/* Line 2: variant + size + ratings */}
                  <div className="flex items-center gap-2 text-[10px] text-muted-foreground/60">
                    {model.variant && <span>{model.variant}</span>}
                    <span>{sizeGB} GB</span>
                    {info && (
                      <>
                        <span className="text-border">|</span>
                        <div className="flex items-center gap-1">
                          <span className="text-[9px]">quality</span>
                          <RatingDots value={info.quality} />
                        </div>
                        <div className="flex items-center gap-1">
                          <span className="text-[9px]">speed</span>
                          <RatingDots value={info.speed} />
                        </div>
                      </>
                    )}
                  </div>
                  {/* Line 3: capability badges (only if info exists and has notable caps) */}
                  {info && (info.capabilities.img2img || info.capabilities.inpaint || info.text_rendering) && (
                    <div className="flex items-center gap-1.5 pt-0.5">
                      {info.capabilities.img2img && (
                        <span className="rounded bg-secondary/60 px-1 py-0.5 text-[8px] text-muted-foreground/60">img2img</span>
                      )}
                      {info.capabilities.inpaint && (
                        <span className="rounded bg-secondary/60 px-1 py-0.5 text-[8px] text-muted-foreground/60">inpaint</span>
                      )}
                      {info.text_rendering && (
                        <span className="rounded bg-primary/10 px-1 py-0.5 text-[8px] font-medium text-primary/70">text</span>
                      )}
                    </div>
                  )}
                </div>
              </SelectItem>
            )
          })}
        </SelectContent>
      </Select>
    </div>
  )
}
