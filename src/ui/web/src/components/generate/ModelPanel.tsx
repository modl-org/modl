import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select'
import type { InstalledModel } from '../../api'
import { modelDefaults, type GenerateFormState } from './generate-state'

type Props = {
  models: InstalledModel[]
  form: GenerateFormState
  setForm: React.Dispatch<React.SetStateAction<GenerateFormState>>
  /** When true, auto-adjust steps/guidance on model change */
  autoDefaults?: boolean
}

export function ModelPanel({ models, form, setForm, autoDefaults = true }: Props) {
  const checkpoints = models.filter((m) => m.model_type === 'checkpoint' || m.model_type === 'diffusion_model')

  const handleChange = (modelId: string) => {
    const model = checkpoints.find((m) => m.id === modelId)
    if (!model) return

    setForm((prev) => {
      const patch: Partial<GenerateFormState> = { base_model_id: modelId }

      if (autoDefaults) {
        const defaults = modelDefaults(model.name)
        patch.steps = defaults.steps
        patch.guidance = defaults.guidance
      }

      return { ...prev, ...patch }
    })
  }

  return (
    <div className="space-y-1.5">
      <span className="text-[11px] font-medium text-muted-foreground">
        Checkpoint
      </span>
      <Select
        value={form.base_model_id}
        onValueChange={handleChange}
        disabled={checkpoints.length === 0}
      >
        <SelectTrigger className="w-full bg-background/60">
          <SelectValue placeholder={checkpoints.length === 0 ? 'No checkpoints installed' : 'Select a model'} />
        </SelectTrigger>
        <SelectContent>
          {checkpoints.map((model) => (
            <SelectItem key={model.id} value={model.id}>
              <div className="flex items-center gap-2">
                <span>{model.name}</span>
                {model.variant && (
                  <span className="text-[10px] text-muted-foreground">({model.variant})</span>
                )}
                <span className="ml-auto text-[10px] text-muted-foreground/60">
                  {(model.size_bytes / 1024 / 1024 / 1024).toFixed(1)}GB
                </span>
              </div>
            </SelectItem>
          ))}
        </SelectContent>
      </Select>
    </div>
  )
}
