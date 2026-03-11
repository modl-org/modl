import { useEffect, useRef } from 'react'

/**
 * Manages an EventSource connection lifecycle.
 *
 * Connects when `url` is non-null, disconnects when `url` becomes null or on unmount.
 * Automatically filters keepalive/idle pings before forwarding to `onMessage`.
 * Uses exponential backoff on reconnect (1s → 2s → 4s → … → 30s cap).
 */
export function useSSE(
  url: string | null,
  onMessage: (data: string) => void,
  options?: { onError?: () => void },
): void {
  const onMessageRef = useRef(onMessage)
  onMessageRef.current = onMessage

  const onErrorRef = useRef(options?.onError)
  onErrorRef.current = options?.onError

  useEffect(() => {
    if (!url) return

    let es: EventSource | null = null
    let backoff = 1000
    let timer: ReturnType<typeof setTimeout> | null = null
    let cancelled = false

    function connect() {
      if (cancelled) return
      es = new EventSource(url!)

      es.onopen = () => {
        backoff = 1000 // reset on successful connection
      }

      es.onmessage = (event) => {
        const data: string = event.data
        if (data === 'keepalive' || data === 'idle') return
        onMessageRef.current(data)
      }

      es.onerror = () => {
        es?.close()
        es = null
        if (cancelled) return

        // Exponential backoff: 1s, 2s, 4s, 8s, 16s, 30s cap
        timer = setTimeout(() => {
          timer = null
          connect()
        }, backoff)
        backoff = Math.min(backoff * 2, 30_000)

        onErrorRef.current?.()
      }
    }

    connect()

    return () => {
      cancelled = true
      es?.close()
      if (timer) clearTimeout(timer)
    }
  }, [url])
}
