import logoUrl from '../../../../assets/logo.png'

export default function BrandMark() {
  return (
    <div className="ds-surface ds-glow-cyan flex h-10 w-10 items-center justify-center rounded-2xl">
      <img src={logoUrl} alt="DigiSteel" className="h-7 w-7 opacity-90" />
    </div>
  )
}

