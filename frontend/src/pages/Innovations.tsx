import { ExternalLink, Layers, Move3d, SquareFunction } from 'lucide-react'
import { Link } from 'react-router-dom'

function StationCard(props: {
  tag: string
  title: string
  description: string
  to: string
  icon: JSX.Element
  accent: 'amber' | 'cyan'
}) {
  const glow = props.accent === 'amber' ? 'ds-glow-amber' : 'ds-glow-cyan'
  const chip = props.accent === 'amber' ? 'text-amber' : 'text-cyan'

  return (
    <Link to={props.to} className={`ds-surface ${glow} block p-6 transition hover:-translate-y-0.5`}>
      <div className="flex items-start justify-between gap-4">
        <div className="min-w-0">
          <div className={`ds-display text-xs tracking-[0.28em] ${chip}`}>{props.tag}</div>
          <h2 className="ds-display mt-3 text-2xl tracking-wide">{props.title}</h2>
          <p className="mt-3 text-fg1">{props.description}</p>
        </div>
        <div className="mt-1 opacity-85">{props.icon}</div>
      </div>
      <div className="mt-6 flex items-center gap-2 text-sm text-fg1">
        <span className="underline decoration-white/20 underline-offset-4">Open in Wiki</span>
        <ExternalLink size={16} className="opacity-75" />
      </div>
    </Link>
  )
}

export default function Innovations() {
  return (
    <div className="mx-auto h-full max-w-6xl px-4 pb-16 pt-8">
      <div className="flex flex-col gap-2">
        <div className="ds-display text-xs tracking-[0.28em] text-muted">INNOVATION BAY</div>
        <h1 className="ds-display text-4xl tracking-wide md:text-5xl">A2 + A3</h1>
        <p className="max-w-[75ch] text-fg1">
          Two small changes that make the model lighter and more robust. This page is the guided
          “machine room” version; the deep technical details live in the wiki.
        </p>
      </div>

      <div className="mt-8 grid gap-6 md:grid-cols-2">
        <StationCard
          tag="A2"
          title="GhostConv Weight-Sharing"
          description="Ghost-style cheap features with shared weights across pyramid stages. The goal: fewer parameters without losing signal."
          to="/wiki/Modules"
          icon={<Move3d size={22} />}
          accent="cyan"
        />
        <StationCard
          tag="A3"
          title="Inner-WIoU Loss"
          description="A composite regression loss: Inner-IoU constraint + WIoU v3 dynamic focusing. The goal: better multi-dataset generalization."
          to="/wiki/API"
          icon={<SquareFunction size={22} />}
          accent="amber"
        />
      </div>

      <div className="mt-8 grid gap-6 md:grid-cols-3">
        <div className="ds-surface p-6">
          <div className="flex items-center gap-2 text-fg1">
            <Layers size={16} />
            <div className="ds-display text-xs tracking-[0.28em] text-muted">NEXT</div>
          </div>
          <div className="mt-4 ds-display text-xl tracking-wide">Factory Tour</div>
          <p className="mt-2 text-fg1">Start at the 3D floor and teleport between stations.</p>
          <Link
            to="/"
            className="mt-5 inline-flex items-center justify-center rounded-full bg-white/10 px-4 py-2 text-sm text-fg0 transition hover:bg-white/15"
          >
            Go to Home
          </Link>
        </div>
        <div className="ds-surface p-6">
          <div className="flex items-center gap-2 text-fg1">
            <SquareFunction size={16} />
            <div className="ds-display text-xs tracking-[0.28em] text-muted">DOCS</div>
          </div>
          <div className="mt-4 ds-display text-xl tracking-wide">Code Wiki</div>
          <p className="mt-2 text-fg1">Architecture, modules, APIs, and running instructions.</p>
          <Link
            to="/wiki"
            className="mt-5 inline-flex items-center justify-center rounded-full bg-white/10 px-4 py-2 text-sm text-fg0 transition hover:bg-white/15"
          >
            Open Wiki
          </Link>
        </div>
        <div className="ds-surface p-6">
          <div className="flex items-center gap-2 text-fg1">
            <Move3d size={16} />
            <div className="ds-display text-xs tracking-[0.28em] text-muted">LAB</div>
          </div>
          <div className="mt-4 ds-display text-xl tracking-wide">Mock Console</div>
          <p className="mt-2 text-fg1">Preview how runs and robustness results will look.</p>
          <Link
            to="/lab"
            className="mt-5 inline-flex items-center justify-center rounded-full bg-white/10 px-4 py-2 text-sm text-fg0 transition hover:bg-white/15"
          >
            Open Lab
          </Link>
        </div>
      </div>
    </div>
  )
}
