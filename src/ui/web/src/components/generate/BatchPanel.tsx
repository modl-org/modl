import { MinusIcon, PlusIcon } from 'lucide-react'
import { Button } from '@/components/ui/button'
import type { GenerateFormState } from './generate-state'

type Props = {
  form: GenerateFormState
  setForm: React.Dispatch<React.SetStateAction<GenerateFormState>>
}

export function BatchPanel({ form, setForm }: Props) {
  const set = (n: number) => {
    if (n >= 1 && n <= 16) setForm((prev) => ({ ...prev, batch_count: n }))
  }

  return (
    <div className="space-y-1.5">
      <span className="text-[11px] font-medium text-muted-foreground">Images</span>
      <div className="flex items-center gap-1.5">
        <Button
          type="button"
          variant="outline"
          size="icon"
          className="size-7"
          disabled={form.batch_count <= 1}
          onClick={() => set(form.batch_count - 1)}
        >
          <MinusIcon className="size-3" />
        </Button>
        <span className="w-6 text-center font-mono text-xs tabular-nums">
          {form.batch_count}
        </span>
        <Button
          type="button"
          variant="outline"
          size="icon"
          className="size-7"
          disabled={form.batch_count >= 16}
          onClick={() => set(form.batch_count + 1)}
        >
          <PlusIcon className="size-3" />
        </Button>
      </div>
    </div>
  )
}
