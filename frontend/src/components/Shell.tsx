import { Outlet, useLocation } from 'react-router-dom'
import TopNav from '@/components/TopNav'
import { cn } from '@/lib/utils'

export default function Shell() {
  const location = useLocation()
  const isHome = location.pathname === '/'

  return (
    <div className="relative h-full w-full overflow-hidden">
      <div
        className={cn(
          'pointer-events-none absolute inset-0 opacity-70',
          isHome ? 'opacity-90' : 'opacity-55',
        )}
        style={{
          background:
            'radial-gradient(900px 520px at 20% 15%, oklch(78% 0.18 65 / 0.14), transparent 60%), radial-gradient(820px 560px at 80% 80%, oklch(82% 0.12 205 / 0.12), transparent 60%)',
        }}
      />
      <TopNav />
      <main className="relative h-full w-full pt-16">
        <Outlet />
      </main>
    </div>
  )
}

