import { useEffect, useMemo } from 'react'
import { Button } from '@/components/ui/button'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Textarea } from '@/components/ui/textarea'
import { type GenerateRequest, type GpuStatus, type InstalledModel } from '../api'
import type { GenerateFormState } from './generate-form-state'
import { GeneratePresets } from './GeneratePresets'
import { LoraStack } from './LoraStack'
import { ModelPicker } from './ModelPicker'

type Props = {
  models: InstalledModel[]
  gpu: GpuStatus
  form: GenerateFormState
  setForm: React.Dispatch<React.SetStateAction<GenerateFormState>>
  onSubmitGenerate: (req: GenerateRequest, expectedCount: number) => Promise<void>
  isSubmitting: boolean
}


export function GenerateForm({
  models,
  gpu,
  form,
  setForm,
  onSubmitGenerate,
  isSubmitting,
}: Props) {
  const checkpoints = models.filter((m) => m.model_type === 'checkpoint')
  const loras = models.filter((m) => m.model_type === 'lora')

  useEffect(() => {
    if (!form.base_model_id && checkpoints.length > 0) {
      setForm((prev) => ({ ...prev, base_model_id: checkpoints[0].id }))
    }
  }, [checkpoints, form.base_model_id, setForm])

  const canSubmit = useMemo(() => {
    return (
      !gpu.training_active &&
      !isSubmitting &&
      form.prompt.trim().length > 0 &&
      form.base_model_id.length > 0
    )
  }, [form.base_model_id, form.prompt, gpu.training_active, isSubmitting])

  const submit = async (event: React.FormEvent) => {
    event.preventDefault()
    if (!canSubmit) return

    const req: GenerateRequest = {
      prompt: form.prompt,
      negative_prompt: form.negative_prompt.trim() ? form.negative_prompt : undefined,
      model_id: form.base_model_id,
      width: form.width,
      height: form.height,
      steps: form.steps,
      guidance: form.guidance,
      seed: form.seed,
      num_images: form.num_images,
      loras: form.loras.map((entry) => ({ id: entry.id, strength: entry.strength })),
    }

    await onSubmitGenerate(req, form.num_images)
  }

  return (
    <Card>
      <CardHeader>
        <CardTitle className="text-base">Generate</CardTitle>
      </CardHeader>
      <CardContent>
        <form className="grid gap-4 xl:grid-cols-[minmax(0,1fr)_320px]" onSubmit={submit}>
          <div className="space-y-4">
            <label className="flex flex-col gap-2">
              <span className="text-xs uppercase tracking-wide text-muted-foreground">Prompt</span>
              <Textarea
                value={form.prompt}
                onChange={(e) => setForm((prev) => ({ ...prev, prompt: e.target.value }))}
                rows={6}
                placeholder="Describe the image you want..."
              />
            </label>

            <ModelPicker
              models={models}
              value={form.base_model_id}
              onChange={(value) => setForm((prev) => ({ ...prev, base_model_id: value }))}
            />

            <LoraStack
              loras={loras}
              value={form.loras}
              onChange={(entries) => setForm((prev) => ({ ...prev, loras: entries }))}
            />

            {checkpoints.length === 0 ? (
              <p className="text-sm text-destructive">Install a checkpoint model first.</p>
            ) : null}
            {gpu.training_active ? (
              <p className="text-sm text-destructive">Generation is locked while training is active.</p>
            ) : null}

            <div className="flex flex-wrap items-center gap-3">
              <Button type="submit" disabled={!canSubmit}>
                {gpu.training_active ? 'GPU busy - training' : isSubmitting ? 'Generating...' : 'Generate'}
              </Button>
              {!gpu.training_active && gpu.vram_free_mb != null ? (
                <span className="text-xs text-muted-foreground">
                  {(gpu.vram_free_mb / 1024).toFixed(1)} GB free
                </span>
              ) : null}
            </div>
          </div>

          <GeneratePresets form={form} setForm={setForm} />
        </form>
      </CardContent>
    </Card>
  )
}
