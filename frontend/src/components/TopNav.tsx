import { Hammer, Layers, ScrollText, Sparkles } from 'lucide-react'
import { NavLink } from 'react-router-dom'
import BrandMark from '@/components/brand/BrandMark'
import { cn } from '@/lib/utils'

function NavItem(props: { to: string; label: string; icon: JSX.Element }) {
  return (
    <NavLink
      to={props.to}
      className={({ isActive }) =>
        cn(
          'group inline-flex items-center gap-2 rounded-full px-3 py-2 text-sm text-fg1 transition',
          'hover:bg-white/5 hover:text-fg0',
          isActive ? 'bg-white/10 text-fg0' : 'bg-transparent',
        )
      }
    >
      <span className="opacity-85 transition group-hover:opacity-100">{props.icon}</span>
      <span className="tracking-wide">{props.label}</span>
    </NavLink>
  )
}

export default function TopNav() {
  return (
    <header className="fixed left-0 right-0 top-0 z-30">
      <div className="mx-auto flex max-w-6xl items-center justify-between px-4 py-3">
        <NavLink to="/" className="group inline-flex items-center gap-3">
          <BrandMark />
          <div className="flex flex-col leading-none">
            <span className="ds-display text-[13px] text-fg0 tracking-[0.18em]">DIGISTEEL</span>
            <span className="text-[12px] text-muted">Factory Skin</span>
          </div>
        </NavLink>

        <nav className="hidden items-center gap-2 md:flex">
          <NavItem to="/innovations" label="Innovations" icon={<Sparkles size={16} />} />
          <NavItem to="/wiki" label="Wiki" icon={<ScrollText size={16} />} />
          <NavItem to="/lab" label="Lab" icon={<Layers size={16} />} />
          <NavItem to="/about" label="About" icon={<Hammer size={16} />} />
        </nav>
      </div>
      <div className="h-px w-full bg-gradient-to-r from-transparent via-white/10 to-transparent" />
    </header>
  )
}
