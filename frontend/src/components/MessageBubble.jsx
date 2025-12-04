function MessageBubble({ role, content, meta }) {
  const palette = {
    user: 'bg-primary text-primary-foreground border-primary/40',
    Researcher: 'bg-blue-900/60 text-blue-50 border-blue-500/40',
    Critic: 'bg-rose-900/50 text-rose-50 border-rose-500/40',
    Experimenter: 'bg-amber-900/50 text-amber-50 border-amber-500/40',
    Analyst: 'bg-emerald-900/50 text-emerald-50 border-emerald-500/40',
    Coordinator: 'bg-cyan-900/50 text-cyan-50 border-cyan-500/40',
    Redacteur: 'bg-indigo-900/50 text-indigo-50 border-indigo-500/40',
    SourceHunterEconTech: 'bg-slate-800 text-slate-100 border-slate-500/50',
    default: 'bg-card text-card-foreground border-border/60',
  }
  const tone = palette[role] || palette.default
  return (
    <div className={`rounded-xl px-4 py-3 shadow-sm border ${tone}`}>
      <div className="text-xs text-muted-foreground mb-1 flex items-center justify-between">
        <span>{meta}</span>
        <span className="text-[11px] uppercase tracking-wide text-muted-foreground/80">{role || 'agent'}</span>
      </div>
      <div className="whitespace-pre-wrap leading-relaxed text-sm">{content}</div>
    </div>
  )
}

export default MessageBubble
