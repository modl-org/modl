import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select'
import type { InstalledModel } from '../api'

type Props = {
  models: InstalledModel[]
  value: string
  onChange: (value: string) => void
}

export function ModelPicker({ models, value, onChange }: Props) {
  const checkpoints = models.filter((m) => m.model_type === 'checkpoint')

  return (
    <label className="flex flex-col gap-2">
      <span className="text-xs uppercase tracking-wide text-muted-foreground">Base model</span>
      <Select
        value={value}
        onValueChange={onChange}
        disabled={checkpoints.length === 0}
      >
        <SelectTrigger className="w-full bg-background">
          <SelectValue placeholder="No checkpoints installed" />
        </SelectTrigger>
        <SelectContent>
          {checkpoints.map((model) => (
            <SelectItem key={model.id} value={model.id}>
              {model.name}
              {model.variant ? ` (${model.variant})` : ''}
            </SelectItem>
          ))}
        </SelectContent>
      </Select>
    </label>
  )
}
