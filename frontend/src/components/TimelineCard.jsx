import MessageBubble from './MessageBubble'

function TimelineCard({ entry }) {
  return (
    <div className="rounded-2xl border border-border bg-card p-4 shadow-sm">
      <div className="flex items-center justify-between mb-2">
        <span className="text-xs text-muted-foreground">Iteration {entry.iteration}</span>
        <span className="text-xs text-muted-foreground">
          {new Date(entry.timestamp || Date.now()).toLocaleTimeString()}
        </span>
      </div>
      <div className="text-sm text-foreground mb-3 whitespace-pre-wrap">{entry.summary || '...'}</div>
      {entry?.messages?.length ? (
        <div className="space-y-2 mb-3">
          {entry.messages.map((msg, idx) => (
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
