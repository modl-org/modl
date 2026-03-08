import { useCallback, useState } from 'react'
import { useMutation, useQueryClient } from '@tanstack/react-query'
import { Camera, Plus } from 'lucide-react'
import { api, type AgentEvent, type StudioSession } from '../../api'
import { useSSE } from '../../hooks/useSSE'
import { AgentTimeline } from './AgentTimeline'
import { IntentInput } from './IntentInput'
import { ResultGallery } from './ResultGallery'
import { SessionHistory } from './SessionHistory'
import { UploadZone } from './UploadZone'

export function StudioView() {
  const queryClient = useQueryClient()

  // Local form state
  const [files, setFiles] = useState<File[]>([])
  const [intent, setIntent] = useState('')

  // Active session
  const [activeSession, setActiveSession] = useState<StudioSession | null>(null)
  const [liveEvents, setLiveEvents] = useState<AgentEvent[]>([])
  const [sseSessionId, setSseSessionId] = useState<string | null>(null)

  // SSE connection — managed by useSSE hook
  const handleStudioMessage = useCallback(
    (data: string) => {
      if (data === 'connected') return

      try {
        const event = JSON.parse(data) as AgentEvent
        setLiveEvents((prev) => [...prev, event])

        // Update session status based on events
        if (event.type === 'output_ready') {
          setActiveSession((prev) =>
            prev
              ? { ...prev, status: 'completed', output_images: event.images ?? [] }
              : null,
          )
          setSseSessionId(null)
          queryClient.invalidateQueries({ queryKey: ['studio-sessions'] })
        } else if (event.type === 'error') {
          setActiveSession((prev) =>
            prev ? { ...prev, status: 'failed' } : null,
          )
        }
      } catch {
        // Non-JSON message — ignore
      }
    },
    [queryClient],
  )

  const handleStudioError = useCallback(() => {
    setSseSessionId(null)
  }, [])

  useSSE(
    sseSessionId ? `/api/studio/sessions/${encodeURIComponent(sseSessionId)}/stream` : null,
    handleStudioMessage,
    { onError: handleStudioError },
  )

  // Create session mutation
  const createSession = useMutation({
    mutationFn: async () => {
      // 1. Create session
      const { id } = await api.studioCreateSession(intent)

      // 2. Upload images
      if (files.length > 0) {
        await api.studioUploadImages(id, files)
      }

      // 3. Start agent
      await api.studioStart(id)

      return id
    },
    onSuccess: (sessionId) => {
      // Clear form
      setFiles([])
      setIntent('')

      // Start SSE stream
      setSseSessionId(sessionId)

      // Set active session with initial state
      setActiveSession({
        id: sessionId,
        intent,
        status: 'running',
        input_images: [],
        output_images: [],
        events: [],
        created_at: new Date().toISOString(),
      })
      setLiveEvents([])

      queryClient.invalidateQueries({ queryKey: ['studio-sessions'] })
    },
  })

  // Load a past session
  const loadSession = useCallback(
    (session: StudioSession) => {
      setSseSessionId(null)
      setActiveSession(session)
      setLiveEvents(session.events)

      // If running, reconnect SSE
      if (session.status === 'running') {
        setSseSessionId(session.id)
      }
    },
    [],
  )

  // Reset to create new session
  const handleNewSession = useCallback(() => {
    setSseSessionId(null)
    setActiveSession(null)
    setLiveEvents([])
    setFiles([])
    setIntent('')
  }, [])

  const isCompleted = activeSession?.status === 'completed'
  const isFailed = activeSession?.status === 'failed'

  return (
    <div className="flex h-full flex-col px-4 py-6 md:px-6 pt-16 md:pt-6 pb-24 md:pb-6">
      <div className="mx-auto w-full max-w-2xl space-y-6">
        {/* Active session view */}
        {activeSession ? (
          <>
            {/* Session header */}
            <div className="flex items-start justify-between gap-4">
              <div>
                <h2 className="text-lg font-semibold text-foreground">
                  {isCompleted && (
                    <span className="mr-2 inline-flex h-5 w-5 items-center justify-center rounded-full bg-emerald-500/20 align-text-bottom">
                      <svg className="h-3 w-3 text-emerald-400" viewBox="0 0 12 12" fill="none">
                        <path d="M2 6l3 3 5-5" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
                      </svg>
                    </span>
                  )}
                  {activeSession.intent}
                </h2>
                {activeSession.input_images.length > 0 && (
                  <p className="mt-0.5 text-xs text-muted-foreground">
                    {activeSession.input_images.length} photos uploaded
                  </p>
                )}
              </div>
              {(isCompleted || isFailed) && (
                <button
                  onClick={handleNewSession}
                  className="inline-flex items-center gap-1.5 rounded-md border border-border/60 px-3 py-1.5 text-xs font-medium text-muted-foreground transition-colors hover:bg-accent hover:text-foreground"
                >
                  <Plus className="h-3.5 w-3.5" />
                  New session
                </button>
              )}
            </div>

            {/* Results gallery (completed) */}
            {isCompleted && activeSession.output_images.length > 0 && (
              <ResultGallery images={activeSession.output_images} />
            )}

            {/* Agent timeline */}
            {liveEvents.length > 0 && (
              <div className="space-y-2">
                {(isCompleted || isFailed) && (
                  <p className="text-xs font-medium text-muted-foreground">Timeline</p>
                )}
                <AgentTimeline events={liveEvents} />
              </div>
            )}

            {/* Error state */}
            {isFailed && (
              <div className="rounded-lg border border-destructive/30 bg-destructive/5 px-4 py-3">
                <p className="text-sm text-destructive">
                  Something went wrong. Try creating a new session.
                </p>
              </div>
            )}
          </>
        ) : (
          /* Empty state — create new session */
          <>
            <div className="text-center">
              <div className="mx-auto mb-4 flex h-12 w-12 items-center justify-center rounded-xl bg-primary/10">
                <Camera className="h-6 w-6 text-primary" />
              </div>
              <h2 className="text-lg font-semibold text-foreground">Studio</h2>
              <p className="mt-1 text-sm text-muted-foreground">
                Upload photos and describe what you want. The AI handles everything.
              </p>
            </div>

            <UploadZone
              files={files}
              onFilesChange={setFiles}
              disabled={createSession.isPending}
            />

            <IntentInput
              value={intent}
              onChange={setIntent}
              disabled={createSession.isPending}
            />

            <button
              onClick={() => createSession.mutate()}
              disabled={
                createSession.isPending || (files.length === 0 && !intent.trim())
              }
              className="w-full rounded-xl bg-primary px-4 py-2.5 text-sm font-medium text-primary-foreground transition-colors hover:bg-primary/90 disabled:opacity-50 disabled:cursor-not-allowed"
            >
              {createSession.isPending ? 'Creating...' : 'Create'}
            </button>

            {createSession.isError && (
              <p className="text-center text-sm text-destructive">
                {createSession.error?.message ?? 'Failed to create session'}
              </p>
            )}
          </>
        )}

        {/* Session history */}
        <div className="border-t border-border/30 pt-4">
          <SessionHistory
            activeSessionId={activeSession?.id}
            onSelectSession={loadSession}
          />
        </div>
      </div>
    </div>
  )
}
