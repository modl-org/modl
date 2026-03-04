import { Database, Images, Sparkles, Zap } from 'lucide-react'
import { useQuery } from '@tanstack/react-query'
import { cn } from '@/lib/utils'
import { api } from '../api'
import type { Tab } from '../App'

const NAV_ITEMS: { id: Tab; label: string; icon: React.ElementType }[] = [
  { id: 'train', label: 'Train', icon: Zap },
  { id: 'generate', label: 'Generate', icon: Sparkles },
  { id: 'outputs', label: 'Outputs', icon: Images },
  { id: 'datasets', label: 'Datasets', icon: Database },
]

type Props = {
  activeTab: Tab
  onTabChange: (tab: Tab) => void
}

export function AppSidebar({ activeTab, onTabChange }: Props) {
  const { data: gpu } = useQuery({
    queryKey: ['gpu'],
    queryFn: api.gpu,
    refetchInterval: 5000,
    staleTime: 4_000,
  })

  const { data: status = [] } = useQuery({
    queryKey: ['status'],
    queryFn: api.status,
    refetchInterval: 2000,
  })

  const activeRun = status.find((r) => r.is_running)
  const vramGB = gpu?.vram_free_mb != null ? (gpu.vram_free_mb / 1024).toFixed(1) : null

  return (
    <aside className="flex h-full w-56 flex-col border-r border-border bg-[#0e0e18] select-none shrink-0">
      {/* Brand */}
      <div className="flex h-14 items-center gap-2.5 border-b border-border px-5">
        <div
          className="flex h-7 w-7 items-center justify-center rounded-lg shrink-0"
          style={{ background: 'linear-gradient(135deg, #7c3aed, #a855f7)' }}
        >
          <Zap className="h-4 w-4 text-white" strokeWidth={2.5} />
        </div>
        <div className="leading-none">
          <span
            className="text-sm font-bold tracking-tight"
            style={{
              background: 'linear-gradient(135deg, #a78bfa, #c084fc)',
              WebkitBackgroundClip: 'text',
              WebkitTextFillColor: 'transparent',
              backgroundClip: 'text',
            }}
          >
            modl
          </span>
          <span className="ml-1.5 rounded bg-primary/15 px-1 py-0.5 text-[10px] font-semibold uppercase tracking-wider text-primary/80">
            preview
          </span>
        </div>
      </div>

      {/* Nav */}
      <nav className="flex-1 overflow-y-auto py-3 px-2 space-y-0.5">
        <p className="px-3 pb-1 pt-2 text-[10px] font-semibold uppercase tracking-widest text-muted-foreground/60">
          Workspace
        </p>
        {NAV_ITEMS.map(({ id, label, icon: Icon }) => {
          const isActive = activeTab === id
          return (
            <button
              key={id}
              onClick={() => onTabChange(id)}
              className={cn(
                'flex w-full items-center gap-3 rounded-md px-3 py-2 text-sm font-medium transition-colors',
                isActive
                  ? 'bg-primary/12 text-primary'
                  : 'text-muted-foreground hover:bg-accent hover:text-foreground',
              )}
            >
              <Icon
                className={cn('h-4 w-4 shrink-0', isActive ? 'text-primary' : 'text-muted-foreground')}
                strokeWidth={isActive ? 2.5 : 2}
              />
              {label}
              {id === 'train' && activeRun && (
                <span className="ml-auto flex h-1.5 w-1.5 shrink-0 animate-pulse rounded-full bg-emerald-400" />
              )}
            </button>
          )
        })}
      </nav>

      {/* GPU status footer */}
      <div className="border-t border-border px-4 py-3">
        {gpu?.training_active ? (
          <div className="flex items-center gap-2 text-xs text-amber-300">
            <span className="h-1.5 w-1.5 shrink-0 animate-pulse rounded-full bg-amber-400" />
            <span className="font-medium">Training active</span>
            {vramGB && (
              <span className="ml-auto text-muted-foreground">{vramGB} GB free</span>
            )}
          </div>
        ) : (
          <div className="flex items-center gap-2 text-xs text-muted-foreground">
            <span className="h-1.5 w-1.5 shrink-0 rounded-full bg-emerald-500" />
            <span>GPU idle</span>
            {vramGB && <span className="ml-auto">{vramGB} GB free</span>}
          </div>
        )}
      </div>
    </aside>
  )
}
