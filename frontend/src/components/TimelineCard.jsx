import MessageBubble from './MessageBubble'

function TimelineCard({ entry }) {
  const entryType = entry.type || (entry.payload && entry.payload.type) || 'update'
  const timestamp = entry.ts || entry.timestamp
  const dateObj =
    typeof timestamp === 'number' ? new Date(timestamp * 1000) : new Date(timestamp || Date.now())
  const dateLabel = dateObj.toLocaleTimeString()
  const summary = entry.summary || (entry.payload && entry.payload.summary) || '...'
  const messages = entry.messages || (entry.payload && entry.payload.messages) || []
  return (
    <div className="rounded-2xl border border-border bg-card p-4 shadow-sm">
      <div className="flex items-center justify-between mb-2">
        <span className="text-xs text-muted-foreground">
          {entry.payload?.iteration || entry.iteration ? `Iteration ${entry.payload?.iteration || entry.iteration}` : 'Timeline'}
        </span>
        <span className="text-xs text-muted-foreground">{dateLabel}</span>
      </div>
      <div className="text-xs text-muted-foreground uppercase tracking-wider mb-2">{entryType}</div>
      <div className="text-sm text-foreground mb-3 whitespace-pre-wrap">{summary}</div>
      {messages.length ? (
        <div className="space-y-2 mb-3">
          {messages.map((msg, idx) => (
            <MessageBubble
              key={idx}
              role={msg.role || msg.agent || 'agent'}
              content={`${msg.tool ? `[${msg.tool}] ` : ''}${msg.content || msg.hypothesis || ''}`}
              meta={msg.agent || msg.role || 'message'}
            />
          ))}
        </div>
      ) : null}
    </div>
  )
}

export default TimelineCard
