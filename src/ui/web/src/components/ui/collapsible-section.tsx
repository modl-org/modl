import * as React from 'react'
import { ChevronDownIcon } from 'lucide-react'
import { cn } from '@/lib/utils'

type Props = {
  title: string
  children: React.ReactNode
  defaultOpen?: boolean
  className?: string
}

export function CollapsibleSection({ title, children, defaultOpen = true, className }: Props) {
  const [open, setOpen] = React.useState(defaultOpen)

  return (
    <div className={cn('border-b border-border/20', className)}>
      <button
        type="button"
        className="flex w-full items-center justify-between px-1 py-3 text-left"
        onClick={() => setOpen((o) => !o)}
      >
        <span className="text-[11px] font-semibold uppercase tracking-wider text-foreground/60">
          {title}
        </span>
        <ChevronDownIcon
          className={cn(
            'size-3.5 text-muted-foreground/50 transition-transform duration-200',
            open && 'rotate-180',
          )}
        />
      </button>
      <div
        className={cn(
          'grid transition-[grid-template-rows] duration-200',
          open ? 'grid-rows-[1fr]' : 'grid-rows-[0fr]',
        )}
      >
        <div className="overflow-hidden">
          <div className="pb-3 pt-0.5">{children}</div>
        </div>
      </div>
    </div>
  )
}
