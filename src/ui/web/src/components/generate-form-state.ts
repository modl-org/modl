import type { LoraEntry } from './LoraStack'

export type GenerateFormState = {
  prompt: string
  negative_prompt: string
  base_model_id: string
  loras: LoraEntry[]
  seed: number
  steps: number
  guidance: number
  width: number
  height: number
  num_images: number
}

export function randomSeed(): number {
  return Math.floor(Math.random() * 4_294_967_295)
}

export function createDefaultGenerateFormState(): GenerateFormState {
  return {
    prompt: '',
    negative_prompt: '',
    base_model_id: '',
    loras: [],
    seed: randomSeed(),
    steps: 28,
    guidance: 7,
    width: 1024,
    height: 1024,
    num_images: 1,
  }
}
