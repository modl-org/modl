import { ChevronsLeft, ChevronsRight, Database, HardDrive, Images, ListIcon, Sparkles, Zap } from 'lucide-react'
import { Tooltip, TooltipContent, TooltipTrigger } from '@/components/ui/tooltip'

function ModlLogo({ size = 28 }: { size?: number }) {
  return (
    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 32 32" fill="none" width={size} height={size}>
      <rect width="32" height="32" rx="6" fill="#7c3aed" />
      <text x="16" y="23" textAnchor="middle" fontFamily="system-ui" fontWeight={800} fontSize="16" fill="white">m</text>
    </svg>
  )
}
import { useQuery } from '@tanstack/react-query'
import { cn } from '@/lib/utils'
import { api, type QueueStatus } from '../api'
import { useGpuStatus } from '../hooks/useGpuStatus'
import type { Tab } from '../App'

const NAV_ITEMS: { id: Tab; label: string; icon: React.ElementType }[] = [
  { id: 'generate', label: 'Generate', icon: Sparkles },
  { id: 'outputs', label: 'Outputs', icon: Images },
  { id: 'datasets', label: 'Datasets', icon: Database },
  { id: 'train', label: 'Train', icon: Zap },
  { id: 'models', label: 'Models', icon: HardDrive },
]

type Props = {
  activeTab: Tab
  onTabChange: (tab: Tab) => void
  collapsed: boolean
  onToggleCollapse: () => void
}

export function AppSidebar({ activeTab, onTabChange, collapsed, onToggleCollapse }: Props) {
  const { data: gpu } = useGpuStatus()

  const { data: status = [] } = useQuery({
    queryKey: ['status'],
    queryFn: api.status,
    refetchInterval: 2000,
  })

  const { data: queueStatus } = useQuery<QueueStatus>({
    queryKey: ['generate-queue'],
    queryFn: api.queueStatus,
    refetchInterval: 2000,
    staleTime: 1500,
  })

  const activeRun = status.find((r) => r.is_running)
  const vramGB = gpu?.vram_free_mb != null ? (gpu.vram_free_mb / 1024).toFixed(1) : null
  const queueJobCount = (queueStatus?.running ? 1 : 0) + (queueStatus?.queue?.length ?? 0)

  return (
    <aside
      className={cn(
        'flex h-full flex-col border-r border-border bg-[#0e0e18] select-none shrink-0 transition-[width] duration-200',
        collapsed ? 'w-14' : 'w-56',
      )}
    >
      {/* Brand */}
      <div className="flex h-14 items-center border-b border-border px-3">
        {collapsed ? (
          <div className="flex w-full justify-center">
            <ModlLogo size={28} />
          </div>
        ) : (
          <div className="flex items-center gap-2.5 px-2">
            <ModlLogo size={28} />
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
        )}
      </div>

      {/* Nav */}
      <nav className="flex-1 overflow-y-auto py-3 px-2 space-y-0.5">
        {!collapsed && (
          <p className="px-3 pb-1 pt-2 text-[10px] font-semibold uppercase tracking-widest text-muted-foreground/60">
            Workspace
          </p>
        )}
        {NAV_ITEMS.map(({ id, label, icon: Icon }) => {
          const isActive = activeTab === id
          const btn = (
            <button
              key={id}
              onClick={() => onTabChange(id)}
              className={cn(
                'relative flex w-full items-center rounded-md text-sm font-medium transition-colors',
                collapsed ? 'justify-center px-0 py-2.5' : 'gap-3 px-3 py-2',
                isActive
                  ? 'bg-primary/12 text-primary'
                  : 'text-muted-foreground hover:bg-accent hover:text-foreground',
              )}
            >
              <Icon
                className={cn('h-4 w-4 shrink-0', isActive ? 'text-primary' : 'text-muted-foreground')}
                strokeWidth={isActive ? 2.5 : 2}
              />
              {!collapsed && label}
              {!collapsed && id === 'train' && activeRun && (
                <span className="ml-auto flex h-1.5 w-1.5 shrink-0 animate-pulse rounded-full bg-emerald-400" />
              )}
              {collapsed && id === 'train' && activeRun && (
                <span className="absolute right-0.5 top-0.5 flex h-1.5 w-1.5 shrink-0 animate-pulse rounded-full bg-emerald-400" />
              )}
            </button>
          )
          if (collapsed) {
            return (
              <Tooltip key={id}>
                <TooltipTrigger asChild>{btn}</TooltipTrigger>
                <TooltipContent side="right" sideOffset={8}>{label}</TooltipContent>
              </Tooltip>
            )
          }
          return btn
        })}
      </nav>

      {/* Queue pill — always visible when jobs are active */}
      {queueJobCount > 0 && (
        <div className="border-t border-border/40 px-2 py-2">
          {collapsed ? (
            <Tooltip>
              <TooltipTrigger asChild>
                <button
                  onClick={() => onTabChange('generate')}
                  className="flex w-full items-center justify-center rounded-md py-2 transition-colors hover:bg-accent"
                >
                  <div className="relative">
                    <ListIcon className="h-4 w-4 text-primary" />
                    <span className="absolute -right-1 -top-1 flex h-3.5 w-3.5 items-center justify-center rounded-full bg-primary text-[8px] font-bold text-primary-foreground">
                      {queueJobCount}
                    </span>
                    <span className="absolute -bottom-0.5 -right-0.5 h-1.5 w-1.5 animate-pulse rounded-full bg-primary" />
                  </div>
                </button>
              </TooltipTrigger>
              <TooltipContent side="right" sideOffset={8}>{queueJobCount} in queue</TooltipContent>
            </Tooltip>
          ) : (
            <button
              onClick={() => onTabChange('generate')}
              className="flex w-full items-center gap-3 rounded-md px-3 py-2 transition-colors hover:bg-accent"
            >
              <div className="relative">
                <ListIcon className="h-4 w-4 text-primary" />
                <span className="absolute -bottom-0.5 -right-0.5 h-1.5 w-1.5 animate-pulse rounded-full bg-primary" />
              </div>
              <span className="text-xs font-medium text-primary">{queueJobCount}</span>
              <span className="text-xs text-muted-foreground">in queue</span>
            </button>
          )}
        </div>
      )}

      {/* Collapse toggle */}
      <div className="border-t border-border/40 px-2 py-2">
        {collapsed ? (
          <Tooltip>
            <TooltipTrigger asChild>
              <button
                onClick={onToggleCollapse}
                className="flex w-full items-center justify-center rounded-md py-2 text-muted-foreground/60 transition-colors hover:bg-accent hover:text-foreground"
              >
                <ChevronsRight className="h-4 w-4" />
              </button>
            </TooltipTrigger>
            <TooltipContent side="right" sideOffset={8}>Expand</TooltipContent>
          </Tooltip>
        ) : (
          <button
            onClick={onToggleCollapse}
            className="flex w-full items-center gap-3 rounded-md px-3 py-2 text-muted-foreground/60 transition-colors hover:bg-accent hover:text-foreground"
          >
            <ChevronsLeft className="h-4 w-4" />
            <span className="text-xs">Collapse</span>
          </button>
        )}
      </div>

      {/* GPU status footer */}
      <div className="border-t border-border px-2 py-3">
        {collapsed ? (
          <Tooltip>
            <TooltipTrigger asChild>
              <div className="flex justify-center">
                <span
                  className={cn(
                    'h-2 w-2 shrink-0 rounded-full',
                    gpu?.training_active ? 'animate-pulse bg-amber-400' : 'bg-emerald-500',
                  )}
                />
              </div>
            </TooltipTrigger>
            <TooltipContent side="right" sideOffset={8}>
              {gpu?.training_active ? 'Training active' : 'GPU idle'}
              {vramGB && ` • ${vramGB} GB free`}
            </TooltipContent>
          </Tooltip>
        ) : gpu?.training_active ? (
          <div className="flex items-center gap-2 px-2 text-xs text-amber-300">
            <span className="h-1.5 w-1.5 shrink-0 animate-pulse rounded-full bg-amber-400" />
            <span className="font-medium">Training active</span>
            {vramGB && (
              <span className="ml-auto text-muted-foreground">{vramGB} GB free</span>
            )}
          </div>
        ) : (
          <div className="flex items-center gap-2 px-2 text-xs text-muted-foreground">
            <span className="h-1.5 w-1.5 shrink-0 rounded-full bg-emerald-500" />
            <span>GPU idle</span>
            {vramGB && <span className="ml-auto">{vramGB} GB free</span>}
          </div>
        )}
      </div>
    </aside>
  )
}
