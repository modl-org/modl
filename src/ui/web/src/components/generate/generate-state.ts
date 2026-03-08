// ---------------------------------------------------------------------------
// Generate form state — single source of truth for the generate view
// ---------------------------------------------------------------------------

export type LoraEntry = {
  id: string
  name: string
  strength: number
}

export type GenerationMode = 'txt2img' | 'img2img'

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
  scheduler: string

  // Size
  width: number
  height: number

  // Batch
  batch_count: number

  // Img2Img
  init_image: string | null // data-url or object-url
  init_image_file: File | null
  denoise_strength: number
}

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
    scheduler: 'euler',
    width: 1024,
    height: 1024,
    batch_count: 1,
    init_image: null,
    init_image_file: null,
    denoise_strength: 0.7,
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

/** Smart defaults based on model type identifiers */
export function modelDefaults(modelName: string): { steps: number; guidance: number } {
  const lower = modelName.toLowerCase()
  if (lower.includes('z-image') || lower.includes('z_image')) {
    return { steps: 8, guidance: 1.0 }
  }
  if (lower.includes('schnell') || lower.includes('turbo') || lower.includes('lightning')) {
    return { steps: 4, guidance: 0.0 }
  }
  if (lower.includes('sdxl')) {
    return { steps: 30, guidance: 7.5 }
  }
  if (lower.includes('sd3') || lower.includes('stable-diffusion-3')) {
    return { steps: 28, guidance: 7.0 }
  }
  // flux-dev and others
  return { steps: 28, guidance: 3.5 }
}
