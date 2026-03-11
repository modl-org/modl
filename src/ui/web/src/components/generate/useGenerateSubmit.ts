import { useCallback, useRef } from 'react'
import { toast } from 'sonner'
import { api, type EditRequest, type GenerateRequest } from '../../api'
import { randomSeed, type GenerateFormState } from './generate-state'
import type { SessionItem } from './SessionStrip'

// ---------------------------------------------------------------------------
// useGenerateSubmit — builds requests and submits generate/edit jobs
// ---------------------------------------------------------------------------

type Deps = {
  form: GenerateFormState
  setForm: React.Dispatch<React.SetStateAction<GenerateFormState>>
  nextSessionId: (prefix: string) => string
  enqueueJob: (item: SessionItem) => boolean
  onSubmitted: (queueLength: number, isFirst: boolean) => void
  markSubmitError: (sessionId: string, error: string, isFirst: boolean) => void
}

export function useGenerateSubmit({
  form,
  setForm,
  nextSessionId,
  enqueueJob,
  onSubmitted,
  markSubmitError,
}: Deps) {
  // Ref to always read the latest form without stale closures
  const formRef = useRef(form)
  formRef.current = form

  // ── Shared HTTP submission ──────────────────────────────────────────
  const submitToServer = useCallback(
    async (
      req: Record<string, unknown>,
      submit: (r: never) => Promise<Response>,
      sessionId: string,
      isFirst: boolean,
      jobType: string,
    ) => {
      try {
        const res = await submit(req as never)
        if (!res.ok) {
          let message = `HTTP ${res.status}`
          try {
            const body = await res.json()
            if (body.error) message = body.error
          } catch {
            const text = await res.text()
            if (text) message = text
          }
          throw new Error(message)
        }
        const body = await res.json()
        onSubmitted(body.queue_length ?? 0, isFirst)
      } catch (err) {
        const message = err instanceof Error ? err.message : String(err)
        console.error(`[${jobType}] submit failed:`, message)
        toast.error(message)
        markSubmitError(sessionId, message, isFirst)
      }
    },
    [onSubmitted, markSubmitError],
  )

  // ── Generate (txt2img / img2img) ────────────────────────────────────
  const handleGenerate = useCallback(async () => {
    const f = formRef.current
    if (!f.prompt.trim() || !f.base_model_id) return

    let seed = f.seed
    if (!f.seed_locked) {
      seed = randomSeed()
      setForm((prev) => ({ ...prev, seed }))
    }

    // Upload init image if img2img
    let initImagePath: string | undefined
    if (f.mode === 'img2img' && f.init_image_file) {
      if (!(f.init_image_file instanceof File)) {
        toast.error('Image reference expired — please re-add the image')
        return
      }
      try {
        const uploaded = await api.upload(f.init_image_file)
        initImagePath = uploaded.path
      } catch (err) {
        toast.error(`Image upload failed: ${err instanceof Error ? err.message : String(err)}`)
        return
      }
    }

    const req: GenerateRequest = {
      prompt: f.prompt,
      negative_prompt: f.negative_prompt.trim() || undefined,
      model_id: f.base_model_id,
      width: f.width,
      height: f.height,
      steps: f.steps,
      guidance: f.guidance,
      seed,
      num_images: f.batch_count,
      loras: f.loras.filter((l) => l.enabled !== false).map((l) => ({ id: l.id, strength: l.strength })),
      init_image: initImagePath,
      strength: initImagePath ? f.denoise_strength : undefined,
      fast: f.fast || undefined,
    }

    const sessionId = nextSessionId('gen')
    const isFirst = enqueueJob({
      id: sessionId,
      status: 'queued',
      prompt: f.prompt,
      model_id: f.base_model_id,
      job_type: 'generate',
      batch_count: f.batch_count,
    })

    console.log('[generate] submitting:', f.base_model_id, f.prompt.slice(0, 60))
    await submitToServer(req, api.generate, sessionId, isFirst, 'generate')
  }, [setForm, nextSessionId, enqueueJob, submitToServer])

  // ── Edit ────────────────────────────────────────────────────────────
  const handleEdit = useCallback(async () => {
    const f = formRef.current
    if (!f.prompt.trim() || !f.base_model_id) return

    const editImages = f.edit_images ?? []
    if (editImages.length === 0) return

    let seed = f.seed
    if (!f.seed_locked) {
      seed = randomSeed()
      setForm((prev) => ({ ...prev, seed }))
    }

    const uploadedPaths: string[] = []
    try {
      for (const img of editImages) {
        if (img.type === 'server') {
          uploadedPaths.push(img.serverPath)
        } else {
          if (!(img.file instanceof File)) {
            toast.error('Image reference expired — please re-add the image')
            return
          }
          const uploaded = await api.upload(img.file)
          uploadedPaths.push(uploaded.path)
        }
      }
    } catch (err) {
      toast.error(`Image upload failed: ${err instanceof Error ? err.message : String(err)}`)
      return
    }

    const req: EditRequest = {
      prompt: f.prompt,
      model_id: f.base_model_id,
      images: uploadedPaths,
      steps: f.steps,
      guidance: f.guidance,
      seed,
      num_images: f.batch_count,
    }

    const sessionId = nextSessionId('edit')
    const isFirst = enqueueJob({
      id: sessionId,
      status: 'queued',
      prompt: f.prompt,
      model_id: f.base_model_id,
      job_type: 'edit',
      batch_count: f.batch_count,
    })

    console.log('[edit] submitting:', f.base_model_id, f.prompt.slice(0, 60))
    await submitToServer(req, api.edit, sessionId, isFirst, 'edit')
  }, [setForm, nextSessionId, enqueueJob, submitToServer])

  // ── Submit based on current mode (for keyboard shortcut) ────────────
  const submitCurrent = useCallback(() => {
    const f = formRef.current
    if (!f.prompt.trim() || !f.base_model_id) return
    if (f.mode === 'edit') {
      handleEdit()
    } else {
      handleGenerate()
    }
  }, [handleGenerate, handleEdit])

  return { handleGenerate, handleEdit, submitCurrent }
}
