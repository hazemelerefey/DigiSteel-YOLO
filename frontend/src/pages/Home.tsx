import { ChevronRight, Play, SkipForward, Square } from 'lucide-react'
import { useMemo } from 'react'
import { useNavigate } from 'react-router-dom'
import { canUseWebGL } from '@/lib/webgl'
import FactoryWorld from '@/scene/FactoryWorld'
import { stations, useFactoryStore, type StationKey } from '@/store/factory'

function FactoryFallback() {
  return (
    <div className="h-full w-full">
      <div className="absolute inset-0 bg-[radial-gradient(circle_at_30%_20%,oklch(82%_0.12_205_/_0.12),transparent_55%),radial-gradient(circle_at_70%_75%,oklch(78%_0.18_65_/_0.10),transparent_55%)]" />
      <div className="absolute inset-0 opacity-70 [background:linear-gradient(90deg,transparent_0,transparent_39px,rgba(255,255,255,0.06)_40px),linear-gradient(transparent_0,transparent_39px,rgba(255,255,255,0.05)_40px)] [background-size:40px_40px]" />
    </div>
  )
}

export default function Home() {
  const navigate = useNavigate()
  const hovered = useFactoryStore((s) => s.hovered)
  const focused = useFactoryStore((s) => s.focused)
  const tour = useFactoryStore((s) => s.tour)
  const startTour = useFactoryStore((s) => s.startTour)
  const stopTour = useFactoryStore((s) => s.stopTour)
  const nextStop = useFactoryStore((s) => s.nextStop)
  const webgl = useMemo(() => (typeof window !== 'undefined' ? canUseWebGL() : false), [])

  const activeKey = focused ?? hovered
  const activeStation = stations.find((s) => s.key === activeKey) ?? null

  const activate = (key: StationKey) => {
    const st = stations.find((s) => s.key === key)
    if (!st) return
    navigate(st.route)
  }

  return (
    <div className="relative h-full w-full">
      <div className="absolute inset-0">
        {webgl ? <FactoryWorld onActivate={activate} /> : <FactoryFallback />}
      </div>

      <div className="pointer-events-none absolute inset-0">
        <div className="mx-auto flex h-full max-w-6xl flex-col justify-between px-4 pb-8 pt-10">
          <div className="pointer-events-auto ds-surface max-w-[560px] p-6 md:p-8">
            <div className="ds-display text-xs tracking-[0.28em] text-muted">DIGISTEEL-YOLO</div>
            <h1 className="ds-display mt-3 text-4xl tracking-wide md:text-5xl">
              Factory World
            </h1>
            <p className="mt-3 max-w-[62ch] text-fg1">
              Walk the floor. Touch the machines. Open the real code wiki from inside the scene.
            </p>

            <div className="mt-6 flex flex-wrap items-center gap-3">
              {!tour ? (
                <button
                  onClick={startTour}
                  className="inline-flex items-center gap-2 rounded-full bg-white/15 px-5 py-2 text-sm text-fg0 transition hover:bg-white/20"
                >
                  <Play size={16} />
                  Start Tour
                </button>
              ) : (
                <button
                  onClick={stopTour}
                  className="inline-flex items-center gap-2 rounded-full bg-white/15 px-5 py-2 text-sm text-fg0 transition hover:bg-white/20"
                >
                  <Square size={16} />
                  Stop Tour
                </button>
              )}

              <button
                onClick={() => navigate('/wiki')}
                className="inline-flex items-center gap-2 rounded-full px-5 py-2 text-sm text-fg1 transition hover:bg-white/10 hover:text-fg0"
              >
                Open Wiki
                <ChevronRight size={16} className="opacity-80" />
              </button>
            </div>
          </div>

          <div className="pointer-events-auto grid gap-4 md:grid-cols-3">
            <div className="ds-surface p-5">
              <div className="ds-display text-xs tracking-[0.28em] text-muted">A2</div>
              <div className="mt-3 ds-display text-lg tracking-wide">GhostConv</div>
              <div className="mt-2 text-sm text-fg1">Weight-sharing across pyramid stages.</div>
            </div>
            <div className="ds-surface p-5">
              <div className="ds-display text-xs tracking-[0.28em] text-muted">A3</div>
              <div className="mt-3 ds-display text-lg tracking-wide">Inner-WIoU</div>
              <div className="mt-2 text-sm text-fg1">Composite loss for generalization.</div>
            </div>
            <div className="ds-surface p-5">
              <div className="ds-display text-xs tracking-[0.28em] text-muted">LAB</div>
              <div className="mt-3 ds-display text-lg tracking-wide">Mock Console</div>
              <div className="mt-2 text-sm text-fg1">A preview of runs and robustness results.</div>
            </div>
          </div>
        </div>
      </div>

      <div className="pointer-events-none absolute bottom-6 left-1/2 w-[min(560px,92vw)] -translate-x-1/2">
        <div className="ds-surface flex items-center justify-between gap-3 px-5 py-3">
          <div className="min-w-0">
            <div className="ds-display text-[11px] tracking-[0.28em] text-muted">
              {tour ? 'GUIDED TOUR' : 'HOVER / CLICK'}
            </div>
            <div className="mt-1 truncate text-sm text-fg1">
              {activeStation ? activeStation.label : 'Select a machine to navigate.'}
            </div>
          </div>

          <div className="pointer-events-auto flex items-center gap-2">
            {tour ? (
              <button
                onClick={nextStop}
                className="inline-flex items-center gap-2 rounded-full bg-white/10 px-4 py-2 text-sm text-fg0 transition hover:bg-white/15"
              >
                <SkipForward size={16} />
                Next
              </button>
            ) : (
              <button
                onClick={() => navigate('/innovations')}
                className="inline-flex items-center gap-2 rounded-full bg-white/10 px-4 py-2 text-sm text-fg0 transition hover:bg-white/15"
              >
                Enter Bay
                <ChevronRight size={16} />
              </button>
            )}
          </div>
        </div>
      </div>
    </div>
  )
}
