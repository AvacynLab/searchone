const api = {
  async startJob(name, query) {
    const res = await fetch(
      `/api/jobs/start?name=${encodeURIComponent(name)}&query=${encodeURIComponent(query)}&max_iterations=2`,
      { method: 'POST' },
    )
    if (!res.ok) throw new Error(`Start failed (${res.status})`)
    return res.json()
  },
  async jobStatus(id) {
    const res = await fetch(`/api/jobs/${id}`)
    if (!res.ok) throw new Error(`Status failed (${res.status})`)
    return res.json()
  },
  async jobTimeline(id) {
    const res = await fetch(`/api/jobs/${id}/timeline`)
    if (!res.ok) throw new Error(`Timeline failed (${res.status})`)
    return res.json()
  },
  async jobOverview(id) {
    const res = await fetch(`/api/jobs/${id}/overview`)
    if (!res.ok) throw new Error(`Overview failed (${res.status})`)
    return res.json()
  },
  async jobDashboard(id) {
    const res = await fetch(`/api/jobs/${id}/diagnostic`)
    if (!res.ok) throw new Error(`Diagnostic failed (${res.status})`)
    return res.json()
  },
  async jobEvidence(id, limit = 50) {
    const res = await fetch(`/api/jobs/${id}/evidence?limit=${limit}`)
    if (!res.ok) throw new Error(`Evidence failed (${res.status})`)
    return res.json()
  },
  async jobMetrics(id) {
    const res = await fetch(`/api/jobs/${id}/metrics`)
    if (!res.ok) throw new Error(`Metrics failed (${res.status})`)
    return res.json()
  },
  async jobDecisions(id, limit = 50) {
    const res = await fetch(`/api/jobs/${id}/decisions?limit=${limit}`)
    if (!res.ok) throw new Error(`Decisions failed (${res.status})`)
    return res.json()
  },
  async listJobs() {
    const res = await fetch(`/api/jobs`)
    if (!res.ok) throw new Error(`Jobs failed (${res.status})`)
    return res.json()
  },
  async jobsBoard(limit = 50, offset = 0) {
    const res = await fetch(`/api/jobs/board?limit=${limit}&offset=${offset}`)
    if (!res.ok) throw new Error(`Jobs board failed (${res.status})`)
    return res.json()
  },
  async setSystemPrompt(prompt) {
    const res = await fetch('/api/prompts/system', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ prompt }),
    })
    if (!res.ok) throw new Error(`System prompt failed (${res.status})`)
    return res.json()
  },
  async setPromptVariant(variant) {
    const res = await fetch('/api/prompts/variant', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ variant }),
    })
    if (!res.ok) throw new Error(`Prompt variant failed (${res.status})`)
    return res.json()
  },
  async getSystemPrompt() {
    const res = await fetch('/api/prompts/system')
    if (!res.ok) throw new Error(`Get system prompt failed (${res.status})`)
    return res.json()
  },
  async getPromptVariant() {
    const res = await fetch('/api/prompts/variant')
    if (!res.ok) throw new Error(`Get prompt variant failed (${res.status})`)
    return res.json()
  },
  async renameJob(id, newName) {
    const res = await fetch(`/api/jobs/${id}/rename?new_name=${encodeURIComponent(newName)}`, { method: 'POST' })
    if (!res.ok) throw new Error(`Rename failed (${res.status})`)
    return res.json()
  },
  async deleteJob(id) {
    const res = await fetch(`/api/jobs/${id}`, { method: 'DELETE' })
    if (!res.ok) throw new Error(`Delete failed (${res.status})`)
    return res.json()
  },
  async retryJob(id, opts = {}) {
    const params = new URLSearchParams()
    if (opts.max_iterations) params.set('max_iterations', opts.max_iterations)
    if (opts.max_duration_seconds) params.set('max_duration_seconds', opts.max_duration_seconds)
    if (opts.max_token_budget) params.set('max_token_budget', opts.max_token_budget)
    const res = await fetch(`/api/jobs/${id}/retry?${params.toString()}`, { method: 'POST' })
    if (!res.ok) throw new Error(`Retry failed (${res.status})`)
    return res.json()
  },
  async searchVector(query, topK = 10) {
    const res = await fetch(`/api/search/vector?q=${encodeURIComponent(query)}&top_k=${topK}`)
    if (!res.ok) throw new Error(`Search failed (${res.status})`)
    return res.json()
  },
  async dashboard() {
    const res = await fetch('/api/dashboard')
    if (!res.ok) throw new Error(`Dashboard failed (${res.status})`)
    return res.json()
  },
  async consolidateKnowledge() {
    const res = await fetch('/api/knowledge/promotions/consolidate', { method: 'POST' })
    if (!res.ok) throw new Error(`Consolidation failed (${res.status})`)
    return res.json()
  },
}

export default api
