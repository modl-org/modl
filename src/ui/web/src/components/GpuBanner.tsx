import { Badge } from '@/components/ui/badge'
import type { GpuStatus } from '../api'

export function GpuBanner({ gpu }: { gpu: GpuStatus }) {
  if (!gpu.training_active) return null

  return (
    <div className="mb-4 flex items-center justify-between rounded-lg border border-amber-500/50 bg-amber-500/10 px-4 py-3 text-sm text-amber-300">
      <span>Training in progress, generation is locked.</span>
      <Badge variant="outline" className="border-amber-500/50 text-amber-300">
        {gpu.vram_free_mb != null ? `${(gpu.vram_free_mb / 1024).toFixed(1)} GB free` : 'GPU busy'}
      </Badge>
    </div>
  )
}
