import { useLocation, useSearch } from 'wouter'
import { type GpuStatus } from './api'
import { AppSidebar } from './components/AppSidebar'
import { DatasetViewer } from './components/DatasetViewer'
import { ModelsView } from './components/ModelsView'
import { GenerateView } from './components/generate'
import { createDefaultGenerateFormState, type GenerateFormState } from './components/generate'
import { GpuBanner } from './components/GpuBanner'
import { MobileNav } from './components/MobileNav'
import { OutputsGallery } from './components/OutputsGallery'
import { QueuePanel } from './components/QueuePanel'
import { TrainingRuns } from './components/TrainingRuns'
import { TrainingStatusBar } from './components/TrainingStatusBar'
import { useGpuStatus } from './hooks/useGpuStatus'
import { useLocalStorage } from './hooks/useLocalStorage'

export type Tab = 'train' | 'generate' | 'outputs' | 'datasets' | 'models'

const PAGE_TITLES: Record<Tab, string> = {
  train: 'Training',
  generate: 'Generate',
  outputs: 'Outputs',
  datasets: 'Datasets',
  models: 'Models',
}

function App() {
  const searchString = useSearch()
  const [, navigate] = useLocation()
  const TABS = ['generate', 'outputs', 'datasets', 'train', 'models'] as const
  const params = new URLSearchParams(searchString)
  const tab: Tab = TABS.find((t) => t === params.get('tab')) ?? 'generate'
  const setTab = (next: Tab) => navigate(`/?tab=${next}`)

  // Sidebar collapsed state (persisted)
  const [sidebarCollapsed, setSidebarCollapsed] = useLocalStorage<boolean>(
    'modl:sidebar-collapsed',
    () => false,
  )

  // Form state — shared between GenerateView and OutputsGallery
  const [form, setForm] = useLocalStorage<GenerateFormState>(
    'modl:generate-form-v2',
    createDefaultGenerateFormState,
  )

  const { data: gpu = { training_active: false } as GpuStatus } = useGpuStatus()

  return (
    <div className="flex h-screen overflow-hidden bg-background">
      {/* Desktop sidebar */}
      <div className="hidden md:flex md:flex-col md:h-full">
        <AppSidebar
          activeTab={tab}
          onTabChange={setTab}
          collapsed={sidebarCollapsed}
          onToggleCollapse={() => setSidebarCollapsed((v) => !v)}
        />
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
          {/* Tab: Train — lazy-mounted */}
          {tab === 'train' && (
            <div className="flex h-full flex-col pb-24 pt-16 md:pb-0 md:pt-0">
              {gpu.training_active && (
                <div className="space-y-4 border-b border-border/50 px-4 py-4 md:px-6">
                  <GpuBanner gpu={gpu} />
                  <TrainingStatusBar />
                </div>
              )}
              <TrainingRuns />
            </div>
          )}

          {/* Tab: Generate — always mounted (primary view, preserves SSE + session state) */}
          <div className={tab === 'generate' ? 'h-full' : 'hidden'}>
            <GenerateView form={form} setForm={setForm} setTab={(t) => setTab(t as Tab)} />
          </div>

          {/* Tab: Outputs — lazy-mounted */}
          {tab === 'outputs' && (
            <div className="px-4 py-6 pb-24 md:px-6 md:pb-6 pt-16 md:pt-6">
              <OutputsGallery setForm={setForm} setActiveTab={setTab} />
            </div>
          )}

          {/* Tab: Datasets — lazy-mounted */}
          {tab === 'datasets' && (
            <div className="px-4 py-6 pb-24 md:px-6 md:pb-6 pt-16 md:pt-6">
              <DatasetViewer />
            </div>
          )}

          {/* Tab: Models — lazy-mounted */}
          {tab === 'models' && (
            <div className="px-4 py-6 pb-24 md:px-6 md:pb-6 pt-16 md:pt-6">
              <ModelsView />
            </div>
          )}
        </main>
      </div>

      {/* Floating generation queue panel — hidden on Generate tab (session strip handles it) */}
      {tab !== 'generate' && <QueuePanel />}
    </div>
  )
}

export default App
