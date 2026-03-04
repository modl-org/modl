import { useEffect, useRef, useState } from 'react'
import { useQuery, useQueryClient } from '@tanstack/react-query'
import { useLocation, useSearch } from 'wouter'
import { toast } from 'sonner'
import { Card, CardContent } from '@/components/ui/card'
import { api, type GenerateRequest, type GpuStatus } from './api'
import { AppSidebar } from './components/AppSidebar'
import { DatasetViewer } from './components/DatasetViewer'
import { GenerateForm } from './components/GenerateForm'
import { createDefaultGenerateFormState, type GenerateFormState } from './components/generate-form-state'
import { GenerateProgress, type GenerateState } from './components/GenerateProgress'
import { GpuBanner } from './components/GpuBanner'
import { MobileNav } from './components/MobileNav'
import { OutputsGallery } from './components/OutputsGallery'
import { TrainingRuns } from './components/TrainingRuns'
import { TrainingStatusBar } from './components/TrainingStatusBar'
import { useLocalStorage } from './hooks/useLocalStorage'

export type Tab = 'train' | 'generate' | 'outputs' | 'datasets'

const PAGE_TITLES: Record<Tab, string> = {
  train: 'Training',
  generate: 'Generate',
  outputs: 'Outputs',
  datasets: 'Datasets',
}

function App() {
  const queryClient = useQueryClient()

  const searchString = useSearch()
  const [, navigate] = useLocation()
  const TABS = ['train', 'generate', 'outputs', 'datasets'] as const
  const params = new URLSearchParams(searchString)
  const tab: Tab = TABS.find((t) => t === params.get('tab')) ?? 'train'
  const setTab = (next: Tab) => navigate(`/?tab=${next}`)

  const [form, setForm] = useLocalStorage<GenerateFormState>(
    'modl:generate-form',
    createDefaultGenerateFormState,
  )
  const [generateState, setGenerateState] = useState<GenerateState>({ status: 'idle' })
  const expectedCountRef = useRef(1)

  const { data: gpu = { training_active: false } as GpuStatus } = useQuery({
    queryKey: ['gpu'],
    queryFn: api.gpu,
    refetchInterval: 5000,
    staleTime: 4_000,
  })

  const {
    data: models = [],
    error: modelError,
    isLoading: modelsLoading,
  } = useQuery({
    queryKey: ['models'],
    queryFn: api.models,
    staleTime: 5 * 60_000,
  })

  const submitGenerate = async (req: GenerateRequest, expectedCount: number) => {
    expectedCountRef.current = expectedCount
    setGenerateState({ status: 'submitting' })

    try {
      const res = await api.generate(req)
      if (!res.ok) {
        const text = await res.text()
        throw new Error(text || `HTTP ${res.status}`)
      }
      setGenerateState({ status: 'streaming', lines: [] })
    } catch (err) {
      const message = err instanceof Error ? err.message : String(err)
      setGenerateState({ status: 'error', message })
    }
  }

  useEffect(() => {
    if (generateState.status !== 'streaming') {
      return
    }

    let done = false
    const finish = (nextState: GenerateState) => {
      if (done) return
      done = true
      setGenerateState(nextState)
    }

    const eventSource = new EventSource('/api/generate/stream')

    eventSource.onmessage = (event) => {
      const message = event.data

      setGenerateState((prev) => {
        if (prev.status !== 'streaming') {
          return prev
        }

        return {
          status: 'streaming',
          lines: [...prev.lines.slice(-59), message],
        }
      })

      const lower = message.toLowerCase()

      if (lower.startsWith('error:')) {
        eventSource.close()
        finish({ status: 'error', message: message.slice(6).trim() || 'Generation failed.' })
        return
      }

      if (lower.includes('completed') || lower.includes('done')) {
        const count = expectedCountRef.current
        eventSource.close()
        finish({ status: 'done', count })
        void queryClient.invalidateQueries({ queryKey: ['outputs'] })
        toast.success(`Generated ${count} image(s).`)
      }
    }

    eventSource.onerror = () => {
      finish({ status: 'error', message: 'Progress stream disconnected.' })
      eventSource.close()
    }

    return () => {
      eventSource.close()
    }
  }, [generateState.status, queryClient])

  return (
    <div className="flex h-screen overflow-hidden bg-background">
      {/* Desktop sidebar */}
      <div className="hidden md:flex md:flex-col md:h-full">
        <AppSidebar activeTab={tab} onTabChange={setTab} />
      </div>

      {/* Mobile nav (top bar + bottom bar + drawer) */}
      <MobileNav activeTab={tab} onTabChange={setTab} />

      {/* Main column */}
      <div className="flex flex-1 flex-col min-w-0 overflow-hidden">
        {/* Desktop page header */}
        <header className="hidden md:flex h-14 shrink-0 items-center gap-4 border-b border-border/50 bg-[#09090e]/90 backdrop-blur px-6">
          <h1 className="text-base font-semibold tracking-tight text-foreground">
            {PAGE_TITLES[tab]}
          </h1>
          {gpu.training_active && (
            <span className="ml-auto flex items-center gap-1.5 rounded-full border border-amber-500/40 bg-amber-500/10 px-3 py-1 text-xs font-medium text-amber-300">
              <span className="h-1.5 w-1.5 animate-pulse rounded-full bg-amber-400" />
              Training active
            </span>
          )}
        </header>

        {/* Scrollable content */}
        <main className="flex-1 overflow-y-auto bg-[#09090e]">
          {/* Tab: Train */}
          {tab === 'train' && (
            <div className="flex flex-col py-6 pb-24 md:pb-6 pt-16 md:pt-6">
              {gpu.training_active && (
                <div className="px-4 md:px-6 mb-4 space-y-4">
                  <GpuBanner gpu={gpu} />
                  <TrainingStatusBar />
                </div>
              )}
              <TrainingRuns />
            </div>
          )}

          {/* Tab: Generate */}
          {tab === 'generate' && (
            <div className="mx-auto max-w-3xl space-y-4 px-4 py-6 pb-24 md:px-6 md:pb-6 pt-16 md:pt-6">
              {modelError ? (
                <Card>
                  <CardContent className="p-4 text-sm text-destructive">
                    Models unavailable: {String(modelError)}
                  </CardContent>
                </Card>
              ) : null}

              {!modelError ? (
                <GenerateForm
                  models={models}
                  gpu={gpu}
                  form={form}
                  setForm={setForm}
                  onSubmitGenerate={submitGenerate}
                  isSubmitting={
                    generateState.status === 'submitting' || generateState.status === 'streaming'
                  }
                />
              ) : null}

              {modelsLoading ? (
                <Card>
                  <CardContent className="p-4 text-sm text-muted-foreground">Loading models…</CardContent>
                </Card>
              ) : null}

              <GenerateProgress
                state={generateState}
                onReset={() => setGenerateState({ status: 'idle' })}
                onViewOutputs={() => {
                  setTab('outputs')
                  setGenerateState({ status: 'idle' })
                }}
              />
            </div>
          )}

          {/* Tab: Outputs */}
          {tab === 'outputs' && (
            <div className="px-4 py-6 pb-24 md:px-6 md:pb-6 pt-16 md:pt-6">
              <OutputsGallery setForm={setForm} setActiveTab={setTab} />
            </div>
          )}

          {/* Tab: Datasets */}
          {tab === 'datasets' && (
            <div className="px-4 py-6 pb-24 md:px-6 md:pb-6 pt-16 md:pt-6">
              <DatasetViewer />
            </div>
          )}
        </main>
      </div>
    </div>
  )
}

export default App
