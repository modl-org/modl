import { useEffect, useState } from 'react'

function resolveInitial<T>(initialValue: T | (() => T)): T {
  return initialValue instanceof Function ? initialValue() : initialValue
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
        return { ...fallback, ...parsed }
      }
      return parsed
    } catch {
      return fallback
    }
  })

  useEffect(() => {
    try {
      window.localStorage.setItem(key, JSON.stringify(storedValue))
    } catch {
      // Ignore write failures (quota/private mode).
    }
  }, [key, storedValue])

  return [storedValue, setStoredValue] as const
}
