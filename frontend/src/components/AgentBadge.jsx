function AgentBadge({ name, role }) {
  return (
    <span className="inline-flex items-center gap-1 rounded-full bg-muted px-3 py-1 text-xs text-muted-foreground">
      <span className="font-semibold text-foreground">{name}</span>
      <span>{role}</span>
    </span>
  )
}

export default AgentBadge
