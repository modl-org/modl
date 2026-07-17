import { createContext, useCallback, useContext, useMemo, type ReactNode } from 'react'
import { useLocation } from 'wouter'
import { useLocalStorage } from '../hooks/useLocalStorage'
import {
  createDefaultGenerateFormState,
  randomSeed,
  type GenerateFormState,
} from '../components/generate/generate-state'
import type { GeneratedImage, LibraryLora } from '../api'
import type { Tab } from '../App'

// ---------------------------------------------------------------------------
// Context
// ---------------------------------------------------------------------------

type FormContextValue = {
  form: GenerateFormState
  setForm: React.Dispatch<React.SetStateAction<GenerateFormState>>
}

const FormContext = createContext<FormContextValue | null>(null)

export function FormProvider({ children }: { children: ReactNode }) {
  const [form, setForm] = useLocalStorage<GenerateFormState>(
    'modl:generate-form-v2',
    createDefaultGenerateFormState,
  )
  // Stable value object — a fresh {form, setForm} per render would re-render
  // every consumer even when the form is unchanged.
  const value = useMemo(() => ({ form, setForm }), [form, setForm])
  return <FormContext.Provider value={value}>{children}</FormContext.Provider>
}

export function useForm(): FormContextValue {
  const ctx = useContext(FormContext)
  if (!ctx) throw new Error('useForm must be used within FormProvider')
  return ctx
}

// ---------------------------------------------------------------------------
// Navigation helpers
// ---------------------------------------------------------------------------

export function useAppNav() {
  const [, navigate] = useLocation()

  const navigateToTab = useCallback(
    (tab: Tab) => navigate(`/?tab=${tab}`),
    [navigate],
  )

  const { form: _form, setForm } = useForm()

  /** Populate the generate form from an existing image and navigate to Generate */
  const useAsRecipe = useCallback(
    (image: GeneratedImage) => {
      setForm((prev) => ({
        ...prev,
        prompt: image.prompt ?? '',
        base_model_id: image.base_model_id ?? prev.base_model_id,
        loras: image.lora_name
          ? [{ id: image.lora_name, name: image.lora_name, strength: image.lora_strength ?? 1.0, enabled: true }]
          : [],
        seed: image.seed ?? randomSeed(),
        steps: image.steps ?? 20,
        guidance: image.guidance ?? 3.5,
        width: image.width ?? 1024,
        height: image.height ?? 1024,
      }))
      navigateToTab('generate')
    },
    [setForm, navigateToTab],
  )

  /** Add a LoRA to the generate form and navigate to Generate */
  const addLoraToForm = useCallback(
    (lora: LibraryLora) => {
      setForm((prev) => {
        const alreadyPresent = prev.loras.some((l) => l.id === lora.name)
        if (alreadyPresent) return prev
        return {
          ...prev,
          loras: [
            ...prev.loras,
            { id: lora.name, name: lora.name, strength: 1.0, enabled: true },
          ],
        }
      })
      navigateToTab('generate')
    },
    [setForm, navigateToTab],
  )

  return { navigateToTab, useAsRecipe, addLoraToForm }
}
