import { Sparkles } from 'lucide-react'

type Props = {
  value: string
  onChange: (value: string) => void
  disabled?: boolean
}

const PLACEHOLDERS = [
  'Studio photoshoot of my dog',
  'Professional headshots',
  'Fantasy art portraits',
  'Product photography on marble',
  'Vintage film style photos',
]

export function IntentInput({ value, onChange, disabled }: Props) {
  const placeholder =
    PLACEHOLDERS[Math.floor(Date.now() / 60000) % PLACEHOLDERS.length]

  return (
    <div className="relative">
      <Sparkles className="absolute left-3 top-3 h-4 w-4 text-muted-foreground" />
      <textarea
        value={value}
        onChange={(e) => onChange(e.target.value)}
        placeholder={placeholder}
        disabled={disabled}
        rows={2}
        className="w-full resize-none rounded-xl border border-border/60 bg-background px-10 py-2.5 text-sm placeholder:text-muted-foreground/50 focus:border-primary focus:outline-none focus:ring-1 focus:ring-primary/30 disabled:opacity-50"
      />
    </div>
  )
}
