import { useQuery } from '@tanstack/react-query'
import { Check, Clock, Loader2, XCircle } from 'lucide-react'
import { cn } from '@/lib/utils'
import { api, type StudioSession } from '../../api'

type Props = {
  activeSessionId?: string
  onSelectSession: (session: StudioSession) => void
}

export function SessionHistory({ activeSessionId, onSelectSession }: Props) {
  const { data: sessions = [] } = useQuery({
    queryKey: ['studio-sessions'],
    queryFn: api.studioSessions,
    refetchInterval: 10_000,
  })

  if (sessions.length === 0) return null

  return (
    <div className="space-y-1.5">
      <p className="px-1 text-[10px] font-semibold uppercase tracking-widest text-muted-foreground/60">
        Past Sessions
      </p>
      {sessions.map((session) => (
        <button
          key={session.id}
          onClick={() => onSelectSession(session)}
          className={cn(
            'flex w-full items-center gap-2 rounded-lg px-3 py-2 text-left text-sm transition-colors',
            activeSessionId === session.id
              ? 'bg-primary/10 text-foreground'
              : 'text-muted-foreground hover:bg-accent hover:text-foreground',
          )}
        >
          <StatusIcon status={session.status} />
          <div className="min-w-0 flex-1">
            <p className="truncate font-medium">{session.intent}</p>
            <p className="text-[10px] text-muted-foreground/60">
              {formatDate(session.created_at)}
              {session.input_images.length > 0 &&
                ` \u00b7 ${session.input_images.length} photos`}
            </p>
          </div>
        </button>
      ))}
    </div>
  )
}

function StatusIcon({ status }: { status: StudioSession['status'] }) {
  switch (status) {
    case 'completed':
      return (
        <div className="flex h-4 w-4 shrink-0 items-center justify-center rounded-full bg-emerald-500/20">
          <Check className="h-2.5 w-2.5 text-emerald-400" />
        </div>
      )
    case 'running':
      return <Loader2 className="h-4 w-4 shrink-0 animate-spin text-primary" />
    case 'failed':
      return <XCircle className="h-4 w-4 shrink-0 text-destructive" />
    default:
      return <Clock className="h-4 w-4 shrink-0 text-muted-foreground/40" />
  }
}

function formatDate(iso: string): string {
  try {
    const d = new Date(iso)
    const now = new Date()
    const diff = now.getTime() - d.getTime()
    const mins = Math.floor(diff / 60_000)
    if (mins < 1) return 'Just now'
    if (mins < 60) return `${mins}m ago`
    const hours = Math.floor(mins / 60)
    if (hours < 24) return `${hours}h ago`
    const days = Math.floor(hours / 24)
    if (days < 7) return `${days}d ago`
    return d.toLocaleDateString()
  } catch {
    return iso
  }
}
