import { Input } from '@/components/ui/input'
import type { GenerateFormState } from './generate-state'

type Props = {
  form: GenerateFormState
  setForm: React.Dispatch<React.SetStateAction<GenerateFormState>>
}

export function BatchPanel({ form, setForm }: Props) {
  return (
    <div className="space-y-1.5">
      <span className="text-[11px] font-medium uppercase tracking-wider text-muted-foreground">
        Batch
      </span>
      <div className="flex items-center gap-2">
        <label className="flex flex-1 flex-col gap-1">
          <span className="text-[10px] text-muted-foreground/70">Count</span>
          <Input
            type="number"
            min={1}
            max={16}
            value={form.batch_count}
            className="h-7 bg-background/60 text-xs"
            onChange={(e) => {
              const n = Number(e.target.value)
              if (Number.isFinite(n) && n >= 1 && n <= 16) {
                setForm((prev) => ({ ...prev, batch_count: Math.floor(n) }))
              }
            }}
          />
        </label>
        {form.batch_count > 1 && (
          <p className="mt-4 text-[10px] text-muted-foreground/50">
            {form.batch_count} images will be generated
          </p>
        )}
      </div>
    </div>
  )
}
