import { Camera, Database, HardDrive, Images, Menu, Sparkles, X, Zap } from 'lucide-react'

function ModlLogo({ size = 24 }: { size?: number }) {
  return (
    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 32 32" fill="none" width={size} height={size}>
      <rect width="32" height="32" rx="6" fill="#7c3aed" />
      <text x="16" y="23" textAnchor="middle" fontFamily="system-ui" fontWeight={800} fontSize="16" fill="white">m</text>
    </svg>
  )
}
import { useState } from 'react'
import { cn } from '@/lib/utils'
import type { Tab } from '../App'
import { AppSidebar } from './AppSidebar'

const NAV_ITEMS: { id: Tab; label: string; icon: React.ElementType }[] = [
  { id: 'studio', label: 'Studio', icon: Camera },
  { id: 'generate', label: 'Generate', icon: Sparkles },
  { id: 'outputs', label: 'Outputs', icon: Images },
  { id: 'datasets', label: 'Datasets', icon: Database },
  { id: 'train', label: 'Train', icon: Zap },
  { id: 'models', label: 'Models', icon: HardDrive },
]

type Props = {
  activeTab: Tab
  onTabChange: (tab: Tab) => void
}

export function MobileNav({ activeTab, onTabChange }: Props) {
  const [open, setOpen] = useState(false)

  return (
    <>
      {/* Top bar */}
      <header className="fixed inset-x-0 top-0 z-40 flex h-14 items-center gap-3 border-b border-border bg-[#0e0e18]/95 backdrop-blur px-4 md:hidden">
        <button
          onClick={() => setOpen((v) => !v)}
          className="flex h-8 w-8 items-center justify-center rounded-md text-muted-foreground hover:bg-accent hover:text-foreground transition-colors"
          aria-label="Toggle menu"
        >
          {open ? <X className="h-5 w-5" /> : <Menu className="h-5 w-5" />}
        </button>
        <div className="flex items-center gap-2">
          <ModlLogo size={24} />
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
          <span className="rounded bg-primary/15 px-1 py-0.5 text-[10px] font-semibold uppercase tracking-wider text-primary/80">
            preview
          </span>
        </div>
      </header>

      {/* Drawer overlay */}
      {open && (
        <div
          className="fixed inset-0 z-30 bg-black/60 md:hidden"
          onClick={() => setOpen(false)}
        />
      )}

      {/* Drawer */}
      <div
        className={cn(
          'fixed inset-y-0 left-0 z-40 w-56 transition-transform duration-200 md:hidden',
          open ? 'translate-x-0' : '-translate-x-full',
        )}
      >
        <AppSidebar
          activeTab={activeTab}
          onTabChange={(tab) => {
            onTabChange(tab)
            setOpen(false)
          }}
          collapsed={false}
          onToggleCollapse={() => {}}
        />
      </div>

      {/* Bottom tab bar (alternative mobile nav) */}
      <nav className="fixed inset-x-0 bottom-0 z-40 flex border-t border-border bg-[#0e0e18]/95 backdrop-blur md:hidden">
        {NAV_ITEMS.map(({ id, label, icon: Icon }) => {
          const isActive = activeTab === id
          return (
            <button
              key={id}
              onClick={() => onTabChange(id)}
              className={cn(
                'flex flex-1 flex-col items-center gap-1 py-2.5 text-[10px] font-medium transition-colors',
                isActive ? 'text-primary' : 'text-muted-foreground',
              )}
            >
              <Icon className="h-4.5 w-4.5" strokeWidth={isActive ? 2.5 : 2} />
              {label}
            </button>
          )
        })}
      </nav>
    </>
  )
}
