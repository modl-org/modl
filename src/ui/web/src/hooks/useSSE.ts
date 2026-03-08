import { useEffect, useRef } from 'react'

/**
 * Manages an EventSource connection lifecycle.
 *
 * Connects when `url` is non-null, disconnects when `url` becomes null or on unmount.
 * Automatically filters keepalive/idle pings before forwarding to `onMessage`.
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

    const es = new EventSource(url)

    es.onmessage = (event) => {
      const data: string = event.data
      if (data === 'keepalive' || data === 'idle') return
      onMessageRef.current(data)
    }

    es.onerror = () => {
      es.close()
      onErrorRef.current?.()
    }

    return () => {
      es.close()
    }
  }, [url])
}
