// ---------------------------------------------------------------------------
// Generate form state — single source of truth for the generate view
// ---------------------------------------------------------------------------

export type LoraEntry = {
  id: string
  name: string
  strength: number
  enabled: boolean
}

export type GenerationMode = 'txt2img' | 'img2img' | 'edit'

export type GenerateFormState = {
  // Mode
  mode: GenerationMode

  // Prompts
  prompt: string
  negative_prompt: string

  // Model
  base_model_id: string

  // LoRAs
  loras: LoraEntry[]

  // Sampling
  steps: number
  guidance: number
  seed: number
  seed_locked: boolean
  scheduler: string

  // Size
  width: number
  height: number

  // Batch
  batch_count: number

  // Fast mode (Lightning distillation LoRA)
  fast: boolean

  // Img2Img
  init_image: string | null // data-url or object-url
  init_image_file: File | null
  denoise_strength: number

  // Edit mode
  edit_images: EditImage[]

  // Video
  num_frames: number
  fps: number
}

export type EditImage =
  | { type: 'file'; preview: string; file: File }           // user-uploaded, needs upload
  | { type: 'server'; preview: string; serverPath: string } // already on server

export function randomSeed(): number {
  return Math.floor(Math.random() * 4_294_967_295)
}

export function createDefaultGenerateFormState(): GenerateFormState {
  return {
    mode: 'txt2img',
    prompt: '',
    negative_prompt: '',
    base_model_id: '',
    loras: [],
    steps: 28,
    guidance: 7,
    seed: randomSeed(),
    seed_locked: false,
    scheduler: 'euler',
    width: 1024,
    height: 1024,
    batch_count: 1,
    fast: false,
    init_image: null,
    init_image_file: null,
    denoise_strength: 0.7,
    edit_images: [],
    num_frames: 121,
    fps: 24,
  }
}

/** Size presets available in the UI */
export const SIZE_PRESETS = [
  { label: '1:1', width: 1024, height: 1024, icon: '■' },
  { label: '3:4', width: 896, height: 1152, icon: '▮' },
  { label: '9:16', width: 768, height: 1344, icon: '▯' },
  { label: '4:3', width: 1152, height: 896, icon: '▬' },
  { label: '16:9', width: 1344, height: 768, icon: '▬▬' },
] as const

export type SizePreset = (typeof SIZE_PRESETS)[number]['label'] | 'custom'

export function detectSizePreset(w: number, h: number): SizePreset {
  const match = SIZE_PRESETS.find((p) => p.width === w && p.height === h)
  return match ? match.label : 'custom'
}

/** Smart defaults from model-families API data.
 *  Falls back to string matching if family data isn't loaded yet. */
export function modelDefaults(
  modelName: string,
  familyInfo?: import('../../api').ModelFamilyInfo | null,
): { steps: number; guidance: number } {
  if (familyInfo) {
    return { steps: familyInfo.default_steps, guidance: familyInfo.default_guidance }
  }
  // Fallback: string matching (for initial load before families are fetched)
  const lower = modelName.toLowerCase()
  if (lower.includes('z-image-turbo') || lower.includes('z_image_turbo')) return { steps: 8, guidance: 1.0 }
  if (lower.includes('klein') || lower.includes('schnell')) return { steps: 4, guidance: 1.0 }
  if (lower.includes('qwen')) return { steps: 25, guidance: 3.0 }
  if (lower.includes('chroma')) return { steps: 25, guidance: 4.0 }
  if (lower.includes('flux2') || lower.includes('flux.2') || lower.includes('flux-2')) return { steps: 28, guidance: 4.0 }
  if (lower.includes('turbo')) return { steps: 4, guidance: 1.0 }
  if (lower.includes('sdxl')) return { steps: 30, guidance: 7.5 }
  if (lower.includes('sd-1.5') || lower.includes('sd15')) return { steps: 30, guidance: 7.5 }
  return { steps: 28, guidance: 3.5 }
}

/** Build a form update that switches to edit mode with the given image.
 *  Looks for qwen-image-edit model and applies its defaults. */
export function buildSendToEdit(
  imageUrl: string,
  serverPath: string,
  models: import('../../api').InstalledModel[],
  families: import('../../api').ModelFamily[],
): (prev: GenerateFormState) => GenerateFormState {
  return (prev) => {
    let modelId = prev.base_model_id
    let steps = prev.steps
    let guidance = prev.guidance
    const editModel = models.find(
      (m) => m.id === 'qwen-image-edit' || m.name.toLowerCase().includes('qwen-image-edit'),
    )
    if (editModel) {
      modelId = editModel.id
      const info = findModelFamily(editModel.name, families)
      const defaults = modelDefaults(editModel.name, info)
      steps = defaults.steps
      guidance = defaults.guidance
    }
    return {
      ...prev,
      mode: 'edit' as const,
      base_model_id: modelId,
      steps,
      guidance,
      edit_images: [{ type: 'server' as const, preview: imageUrl, serverPath }],
    }
  }
}

/** Find the ModelFamilyInfo that best matches a model name. */
export function findModelFamily(
  modelName: string,
  families: import('../../api').ModelFamily[],
): import('../../api').ModelFamilyInfo | null {
  const lower = modelName.toLowerCase()
  for (const family of families) {
    for (const model of family.models) {
      if (lower === model.id || lower.includes(model.id)) {
        return model
      }
    }
  }
  // Fuzzy: check arch_key
  for (const family of families) {
    for (const model of family.models) {
      if (lower.includes(model.arch_key.replace(/_/g, '-'))) {
        return model
      }
    }
  }
  return null
}
