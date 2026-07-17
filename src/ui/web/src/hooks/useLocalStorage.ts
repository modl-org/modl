import { useEffect, useState } from 'react'

function resolveInitial<T>(initialValue: T | (() => T)): T {
  return initialValue instanceof Function ? initialValue() : initialValue
}

/**
 * Keys whose values cannot survive JSON round-tripping (File objects, Blobs, etc.).
 * They are stripped before writing and reset to defaults on read.
 */
const NON_SERIALIZABLE_KEYS = new Set(['init_image_file', 'init_image'])

/**
 * Sanitize edit_images: keep 'server' entries (all strings, serializable),
 * strip 'file' entries (contain File objects that can't survive JSON).
 */
function sanitizeEditImages(images: unknown): unknown {
  if (!Array.isArray(images)) return []
  return images.filter(
    (img) => typeof img === 'object' && img !== null && (img as Record<string, unknown>).type === 'server',
  )
}

export function useLocalStorage<T>(key: string, initialValue: T | (() => T)) {
  const [storedValue, setStoredValue] = useState<T>(() => {
    const fallback = resolveInitial(initialValue)

    if (typeof window === 'undefined') {
      return fallback
    }

    try {
      const item = window.localStorage.getItem(key)
      if (!item) return fallback
      const parsed = JSON.parse(item) as T
      // Merge with defaults so new fields are always present
      if (typeof fallback === 'object' && fallback !== null && !Array.isArray(fallback)) {
        const merged = { ...fallback, ...parsed }
        // Restore non-serializable fields to their defaults
        for (const k of NON_SERIALIZABLE_KEYS) {
          if (k in (fallback as Record<string, unknown>)) {
            ;(merged as Record<string, unknown>)[k] = (fallback as Record<string, unknown>)[k]
          }
        }
        return merged
      }
      return parsed
    } catch {
      return fallback
    }
  })

  // Debounced write — serializing the whole form synchronously on every
  // keystroke costs a visible chunk of the frame budget.
  useEffect(() => {
    const timer = window.setTimeout(() => {
      try {
        let toStore: unknown = storedValue
        // Strip non-serializable fields before writing
        if (typeof storedValue === 'object' && storedValue !== null && !Array.isArray(storedValue)) {
          const copy = { ...storedValue } as Record<string, unknown>
          for (const k of NON_SERIALIZABLE_KEYS) {
            delete copy[k]
          }
          // Keep server-type edit_images, strip file-type ones
          if ('edit_images' in copy) {
            copy.edit_images = sanitizeEditImages(copy.edit_images)
          }
          toStore = copy
        }
        window.localStorage.setItem(key, JSON.stringify(toStore))
      } catch {
        // Ignore write failures (quota/private mode).
      }
    }, 400)
    return () => window.clearTimeout(timer)
  }, [key, storedValue])

  return [storedValue, setStoredValue] as const
}
