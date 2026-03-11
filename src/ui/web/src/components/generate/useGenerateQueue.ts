import { useCallback, useRef, useState } from 'react'
import { useQueryClient } from '@tanstack/react-query'
import { toast } from 'sonner'
import { api, type GeneratedImage, type GeneratedOutput } from '../../api'
import { useSSE } from '../../hooks/useSSE'
import type { GenerateProgressState } from './GenerateProgressBar'
import type { PreviewImage } from './ImagePreview'
import type { SessionItem } from './SessionStrip'

// ---------------------------------------------------------------------------
// useGenerateQueue — manages SSE stream, session items, preview, and progress
// ---------------------------------------------------------------------------

export function useGenerateQueue() {
  const queryClient = useQueryClient()

  const [progressState, setProgressState] = useState<GenerateProgressState>({ status: 'idle' })
  const [previewImages, setPreviewImages] = useState<PreviewImage[]>([])
  const [sessionItems, setSessionItems] = useState<SessionItem[]>([])
  const [activeSessionId, setActiveSessionId] = useState<string | null>(null)
  const [sseConnected, setSseConnected] = useState(false)

  const expectedCountRef = useRef(1)
  const userFocusedRef = useRef(false)
  const sessionIdCounter = useRef(0)
  const sessionItemsRef = useRef(sessionItems)
  sessionItemsRef.current = sessionItems
  const isGeneratingRef = useRef(false)

  const isGenerating = progressState.status === 'submitting' || progressState.status === 'streaming'
  isGeneratingRef.current = isGenerating

  // ── Promote next queued → active, or close SSE ──────────────────────
  function promoteOrClose(items: SessionItem[]): SessionItem[] {
    const nextQueued = items.findIndex((s) => s.status === 'queued')
    if (nextQueued !== -1) {
      items[nextQueued] = { ...items[nextQueued], status: 'active' }
    } else {
      setSseConnected(false)
    }
    return items
  }

  // ── SSE message handler ─────────────────────────────────────────────
  const handleSSEMessage = useCallback(
    (message: string) => {
      if (message === 'running' || message.startsWith('running:queue:')) return
      const lower = message.toLowerCase()

      // Queue status pings
      if (lower === 'queue:empty' || lower.startsWith('queue:')) return

      // Error from server
      if (lower.startsWith('error:')) {
        const errMsg = message.slice(6).trim() || 'Generation failed.'
        console.error('[generate] server error:', errMsg)
        toast.error(errMsg)
        setProgressState({ status: 'error', message: errMsg })
        setSessionItems((prev) => {
          const idx = prev.findIndex((s) => s.status === 'active')
          if (idx === -1) return prev
          const updated = [...prev]
          updated[idx] = { ...updated[idx], status: 'error', error: errMsg }
          return promoteOrClose(updated)
        })
        return
      }

      // Completion
      if (lower.includes('completed') || lower.includes('done')) {
        const count = expectedCountRef.current

        void queryClient.invalidateQueries({ queryKey: ['outputs'] }).then(() => {
          const outputs = queryClient.getQueryData<GeneratedOutput[]>(['outputs']) ?? []
          const allImages: Array<{ url: string; seed?: number; modified: number; path?: string }> = []
          for (const group of outputs) {
            for (const img of group.images) {
              allImages.push({ url: `/files/${img.path}`, seed: img.seed, modified: img.modified, path: img.path })
            }
          }
          allImages.sort((a, b) => b.modified - a.modified)
          const completedImages = allImages.slice(0, count)

          if (!userFocusedRef.current) {
            setPreviewImages(completedImages)
          }

          setSessionItems((prev) => {
            const idx = prev.findIndex((s) => s.status === 'active')
            if (idx === -1) return prev
            const updated = [...prev]
            updated[idx] = {
              ...updated[idx],
              status: 'completed',
              images: completedImages.map((ci) => ({ url: ci.url, seed: ci.seed, path: ci.path })),
              step: undefined,
              totalSteps: undefined,
            }
            if (!userFocusedRef.current) {
              setActiveSessionId(updated[idx].id)
            }
            return promoteOrClose(updated)
          })
        })

        toast.success(`Generated ${count} image${count !== 1 ? 's' : ''}`)
        setProgressState({ status: 'done', count, images: [] })
        return
      }

      // Streaming log lines
      setProgressState((prev) => {
        if (prev.status !== 'streaming' && prev.status !== 'submitting') return prev
        const lines = prev.status === 'streaming' ? [...prev.lines.slice(-59), message] : [message]
        return {
          status: 'streaming',
          lines,
          step: prev.status === 'streaming' ? prev.step : undefined,
          totalSteps: prev.status === 'streaming' ? prev.totalSteps : undefined,
        }
      })

      // Parse structured step progress
      try {
        const parsed = JSON.parse(message)
        if (parsed.step != null && parsed.total_steps != null) {
          setProgressState((prev) => {
            if (prev.status !== 'streaming') return prev
            return { ...prev, step: parsed.step, totalSteps: parsed.total_steps }
          })
          setSessionItems((prev) => {
            const idx = prev.findIndex((s) => s.status === 'active')
            if (idx === -1) return prev
            const updated = [...prev]
            updated[idx] = { ...updated[idx], step: parsed.step, totalSteps: parsed.total_steps }
            return updated
          })
        }
      } catch {
        // Not JSON — raw log line, already handled
      }
    },
    [queryClient],
  )

  const handleSSEError = useCallback(() => {
    console.warn('[generate] SSE stream disconnected')
    setSseConnected(false)
    setProgressState((prev) =>
      prev.status === 'streaming' || prev.status === 'submitting'
        ? { status: 'error', message: 'Progress stream disconnected.' }
        : prev,
    )
  }, [])

  useSSE(sseConnected ? '/api/generate/stream' : null, handleSSEMessage, {
    onError: handleSSEError,
  })

  // ── Actions ─────────────────────────────────────────────────────────

  const nextSessionId = useCallback((prefix: string) => {
    return `${prefix}-${++sessionIdCounter.current}-${Date.now()}`
  }, [])

  /** Add a job to the queue. Returns true if this is the first (active) job. */
  const enqueueJob = useCallback((item: SessionItem): boolean => {
    const isFirst = !isGeneratingRef.current
    if (isFirst) {
      expectedCountRef.current = item.batch_count
      setPreviewImages([])
      setProgressState({ status: 'submitting' })
    }
    setSessionItems((prev) => [...prev, { ...item, status: isFirst ? 'active' : 'queued' }])
    userFocusedRef.current = false
    setSseConnected(true)
    return isFirst
  }, [])

  /** Called after successful HTTP submission */
  const onSubmitted = useCallback((queueLength: number, isFirst: boolean) => {
    if (queueLength > 0) {
      toast.info(`Enqueued — position ${queueLength}`)
    } else if (isFirst) {
      setProgressState({ status: 'streaming', lines: [] })
    }
  }, [])

  /** Mark a job submission as failed */
  const markSubmitError = useCallback((sessionId: string, error: string, isFirst: boolean) => {
    setSessionItems((prev) =>
      prev.map((s) => (s.id === sessionId ? { ...s, status: 'error' as const, error } : s)),
    )
    if (isFirst) {
      setProgressState({ status: 'error', message: error })
    }
  }, [])

  const cancelQueued = useCallback(async (sessionId: string) => {
    const items = sessionItemsRef.current
    const queuedItems = items.filter((s) => s.status === 'queued')
    const queueIdx = queuedItems.findIndex((s) => s.id === sessionId)
    if (queueIdx >= 0) {
      await api.cancelQueueItem(queueIdx)
    }
    setSessionItems((prev) => prev.filter((s) => s.id !== sessionId))
  }, [])

  const clearQueue = useCallback(async () => {
    await api.clearQueue()
    setSessionItems((prev) => prev.filter((s) => s.status !== 'queued'))
    toast.info('Queue cleared')
  }, [])

  const selectSession = useCallback((item: SessionItem) => {
    userFocusedRef.current = true
    setActiveSessionId(item.id)
    if (item.status === 'completed' && item.images && item.images.length > 0) {
      setPreviewImages(item.images.map((img) => ({ url: img.url, seed: img.seed })))
    }
    if (item.status === 'error' && item.error) {
      toast.error(item.error)
    }
  }, [])

  const selectGalleryImage = useCallback((img: GeneratedImage) => {
    userFocusedRef.current = true
    setActiveSessionId(null)
    setPreviewImages([{ url: `/files/${img.path}`, seed: img.seed }])
  }, [])

  const clearPreview = useCallback(() => {
    setPreviewImages([])
    setActiveSessionId(null)
  }, [])

  const interrupt = useCallback(() => {
    setProgressState({ status: 'idle' })
  }, [])

  return {
    sessionItems,
    activeSessionId,
    previewImages,
    progressState,
    isGenerating,
    nextSessionId,
    enqueueJob,
    onSubmitted,
    markSubmitError,
    cancelQueued,
    clearQueue,
    selectSession,
    selectGalleryImage,
    clearPreview,
    interrupt,
  }
}
