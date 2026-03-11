import { useCallback, useState } from 'react'
import { MinusIcon, PlusIcon, SparklesIcon, LoaderCircleIcon } from 'lucide-react'
import { Textarea } from '@/components/ui/textarea'
import { cn } from '@/lib/utils'
import { api, type EnhanceRequest } from '../../api'
import type { GenerateFormState } from './generate-state'

type Props = {
  form: GenerateFormState
  setForm: React.Dispatch<React.SetStateAction<GenerateFormState>>
  showNegative?: boolean
  /** Model name hint for model-specific enhancement */
  modelHint?: string
  /** Placeholder override for the prompt textarea */
  placeholder?: string
}

export function PromptPanel({ form, setForm, showNegative = true, modelHint, placeholder }: Props) {
  const [isEnhancing, setIsEnhancing] = useState(false)
  const [negOpen, setNegOpen] = useState(!!form.negative_prompt.trim())

  const handleEnhance = useCallback(async () => {
    if (!form.prompt.trim() || isEnhancing) return

    setIsEnhancing(true)
    try {
      const req: EnhanceRequest = {
        prompt: form.prompt,
        model_hint: modelHint,
        intensity: 'moderate',
      }
      const result = await api.enhance(req)
      setForm((prev) => ({ ...prev, prompt: result.enhanced }))
    } catch (err) {
      console.error('Enhance failed:', err)
    } finally {
      setIsEnhancing(false)
    }
  }, [form.prompt, modelHint, isEnhancing, setForm])

  return (
    <div className="space-y-2 pb-3 border-b border-border/20">
      {/* Positive prompt */}
      <div className="flex flex-col gap-1.5">
        <span className="text-xs font-semibold text-foreground/90">
          Prompt
        </span>
        <div className="relative">
          <Textarea
            value={form.prompt}
            onChange={(e) => setForm((prev) => ({ ...prev, prompt: e.target.value }))}
            placeholder={placeholder ?? "Describe the image you want to create..."}
            className="min-h-[3.5rem] resize-y bg-background/60 pb-8 font-mono text-sm leading-relaxed placeholder:text-muted-foreground/40"
          />
          {/* Enhance — inline pill at bottom-right of textarea */}
          <button
            type="button"
            disabled={!form.prompt.trim() || isEnhancing}
            onClick={handleEnhance}
            title="Rewrite prompt with AI to improve image quality"
            className="absolute bottom-2 right-2 flex items-center gap-1 rounded-full border border-border/40 bg-background/80 px-2 py-0.5 text-[10px] font-medium text-muted-foreground backdrop-blur transition-colors hover:border-primary/40 hover:text-primary disabled:pointer-events-none disabled:opacity-40"
          >
            {isEnhancing ? (
              <LoaderCircleIcon className="size-3 animate-spin" />
            ) : (
              <SparklesIcon className="size-3" />
            )}
            Enhance
          </button>
        </div>
      </div>

      {/* Negative prompt — collapsible */}
      {showNegative && (
        <div>
          <button
            type="button"
            className="flex w-full items-center gap-1.5 py-1 text-left"
            onClick={() => setNegOpen((o) => !o)}
          >
            {negOpen ? (
              <MinusIcon className="size-3 text-muted-foreground/60" />
            ) : (
              <PlusIcon className="size-3 text-muted-foreground/60" />
            )}
            <span className="text-[11px] font-medium text-muted-foreground/70">
              Negative prompt
            </span>
          </button>
          <div
            className={cn(
              'grid transition-[grid-template-rows] duration-200',
              negOpen ? 'grid-rows-[1fr]' : 'grid-rows-[0fr]',
            )}
          >
            <div className="overflow-hidden">
              <Textarea
                value={form.negative_prompt}
                onChange={(e) =>
                  setForm((prev) => ({ ...prev, negative_prompt: e.target.value }))
                }
                rows={2}
                placeholder="Things to avoid (e.g. blurry, low quality, watermark)..."
                className="mt-1 resize-y bg-background/60 text-sm placeholder:text-muted-foreground/30"
              />
            </div>
          </div>
        </div>
      )}
    </div>
  )
}
