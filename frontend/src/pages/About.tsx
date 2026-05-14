import bannerUrl from '../../../assets/banner.png'

export default function About() {
  return (
    <div className="mx-auto h-full max-w-6xl px-4 pb-16 pt-8">
      <div className="ds-surface overflow-hidden">
        <div className="relative">
          <img src={bannerUrl} alt="DigiSteel-YOLO" className="h-40 w-full object-cover opacity-70" />
          <div className="absolute inset-0 bg-gradient-to-b from-black/20 via-black/35 to-black/70" />
          <div className="absolute bottom-0 left-0 right-0 p-6 md:p-8">
            <div className="ds-display text-xs tracking-[0.28em] text-muted">CREDITS</div>
            <h1 className="ds-display mt-2 text-3xl tracking-wide md:text-4xl">
              DigiSteel-YOLO
            </h1>
            <p className="mt-2 max-w-[72ch] text-fg1">
              Robust real-time steel surface defect detection using lightweight YOLO models.
            </p>
          </div>
        </div>

        <div className="grid gap-6 p-6 md:grid-cols-2 md:gap-8 md:p-8">
          <section>
            <div className="ds-display text-xs tracking-[0.28em] text-muted">TEAM</div>
            <ul className="mt-4 space-y-2 text-fg1">
              <li>Hazem Elerefy (Lead)</li>
              <li>Youssef Sherif</li>
              <li>Mohamed Salah</li>
              <li>Moamen Esmat</li>
              <li>Mahmoud Hisham</li>
            </ul>
          </section>

          <section>
            <div className="ds-display text-xs tracking-[0.28em] text-muted">SUPERVISION</div>
            <div className="mt-4 text-fg1">Dr. Tarek Ghoneimy</div>
            <div className="mt-8 ds-display text-xs tracking-[0.28em] text-muted">LICENSE</div>
            <div className="mt-4 text-fg1">MIT</div>
          </section>
        </div>
      </div>
    </div>
  )
}

