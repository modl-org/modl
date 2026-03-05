import { useCallback, useState } from 'react'
import { SparklesIcon, LoaderCircleIcon } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { Textarea } from '@/components/ui/textarea'
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select'
import { api, type EnhanceRequest } from '../../api'
import type { GenerateFormState } from './generate-state'

type EnhanceIntensity = 'subtle' | 'moderate' | 'aggressive'

type Props = {
  form: GenerateFormState
  setForm: React.Dispatch<React.SetStateAction<GenerateFormState>>
  showNegative?: boolean
  /** Model name hint for model-specific enhancement */
  modelHint?: string
}

export function PromptPanel({ form, setForm, showNegative = true, modelHint }: Props) {
  const [isEnhancing, setIsEnhancing] = useState(false)
  const [intensity, setIntensity] = useState<EnhanceIntensity>('moderate')

  const handleEnhance = useCallback(async () => {
    if (!form.prompt.trim() || isEnhancing) return

    setIsEnhancing(true)
    try {
      const req: EnhanceRequest = {
        prompt: form.prompt,
        model_hint: modelHint,
        intensity,
      }
      const result = await api.enhance(req)
      setForm((prev) => ({ ...prev, prompt: result.enhanced }))
    } catch (err) {
      console.error('Enhance failed:', err)
    } finally {
      setIsEnhancing(false)
    }
  }, [form.prompt, modelHint, intensity, isEnhancing, setForm])

  return (
    <div className="space-y-3">
      {/* Positive prompt */}
      <div className="flex flex-col gap-1.5">
        <div className="flex items-center justify-between">
          <span className="text-[11px] font-medium uppercase tracking-wider text-muted-foreground">
            Prompt
          </span>
          <div className="flex items-center gap-1.5">
            <Select value={intensity} onValueChange={(v) => setIntensity(v as EnhanceIntensity)}>
              <SelectTrigger className="h-6 w-[90px] border-0 bg-transparent px-1.5 text-[10px] text-muted-foreground/60 shadow-none">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="subtle">Subtle</SelectItem>
                <SelectItem value="moderate">Moderate</SelectItem>
                <SelectItem value="aggressive">Aggressive</SelectItem>
              </SelectContent>
            </Select>
            <Button
              type="button"
              variant="ghost"
              size="sm"
              className="h-6 gap-1 px-2 text-[10px] text-muted-foreground hover:text-primary"
              disabled={!form.prompt.trim() || isEnhancing}
              onClick={handleEnhance}
              title="Enhance prompt with AI"
            >
              {isEnhancing ? (
                <LoaderCircleIcon className="size-3 animate-spin" />
              ) : (
                <SparklesIcon className="size-3" />
              )}
              Enhance
            </Button>
          </div>
        </div>
        <Textarea
          value={form.prompt}
          onChange={(e) => setForm((prev) => ({ ...prev, prompt: e.target.value }))}
          rows={4}
          placeholder="Describe the image you want to create..."
          className="resize-y bg-background/60 font-mono text-sm leading-relaxed placeholder:text-muted-foreground/50"
        />
      </div>

      {/* Negative prompt */}
      {showNegative && (
        <label className="flex flex-col gap-1.5">
          <span className="text-[11px] font-medium uppercase tracking-wider text-muted-foreground">
            Negative prompt
          </span>
          <Textarea
            value={form.negative_prompt}
            onChange={(e) => setForm((prev) => ({ ...prev, negative_prompt: e.target.value }))}
            rows={2}
            placeholder="Things to avoid (e.g. blurry, low quality, watermark)..."
            className="resize-y bg-background/60 text-sm placeholder:text-muted-foreground/40"
          />
        </label>
      )}
    </div>
  )
}
