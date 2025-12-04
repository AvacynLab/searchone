import React, { useEffect, useMemo, useRef, useState } from 'react'
import * as Dialog from '@radix-ui/react-dialog'
import api from '../api/client'
import MessageBubble from '../components/MessageBubble'
import TimelineCard from '../components/TimelineCard'
import { logClientEvent } from '../utils/logger'
import { captureClientError, initSentry } from '../utils/sentry'

export default function App() {
  const [input, setInput] = useState('')
  const [messages, setMessages] = useState([])
  const [toolOutputs, setToolOutputs] = useState([])
  const [activeJob, setActiveJob] = useState(null)
  const [jobStatus, setJobStatus] = useState(null)
  const [jobOverview, setJobOverview] = useState(null)
  const [timeline, setTimeline] = useState([])
  const [evidence, setEvidence] = useState([])
  const [jobsHistory, setJobsHistory] = useState([])
  const [showHistory, setShowHistory] = useState(false)
  const [error, setError] = useState('')
  const [notice, setNotice] = useState('')
  const [sseError, setSseError] = useState('')
  const [showPromptDialog, setShowPromptDialog] = useState(false)
  const [promptText, setPromptText] = useState('')
  const [promptVariant, setPromptVariant] = useState('')
  const [showNewDialog, setShowNewDialog] = useState(false)
  const [newQuery, setNewQuery] = useState('')
  const [showRenameDialog, setShowRenameDialog] = useState(false)
  const [renameTarget, setRenameTarget] = useState(null)
  const [renameValue, setRenameValue] = useState('')
  const [showDeleteDialog, setShowDeleteDialog] = useState(false)
  const [deleteTarget, setDeleteTarget] = useState(null)
  const [showRetryDialog, setShowRetryDialog] = useState(false)
  const [retryTarget, setRetryTarget] = useState(null)
  const [retryIterations, setRetryIterations] = useState(5)
  const [retryDuration, setRetryDuration] = useState(300)
  const [retryBudget, setRetryBudget] = useState(0)
  const [vectorQuery, setVectorQuery] = useState('')
  const [vectorResults, setVectorResults] = useState([])
  const [vectorLoading, setVectorLoading] = useState(false)
  const [dashboard, setDashboard] = useState(null)
  const [dashboardLoading, setDashboardLoading] = useState(false)
  const [sidebarOpen, setSidebarOpen] = useState(true)
  const [diagnostic, setDiagnostic] = useState(null)
  const diagnosticItems = useMemo(() => {
    const diag = diagnostic?.diagnostic || {}
    const metrics = jobOverview?.run_metrics || {}
    const out = []
    const push = (label, value) => {
      if (value !== undefined && value !== null && value !== '') out.push({ label, value })
    }
    push('coverage', metrics.coverage_score ?? diag.coverage)
    push('evidence', metrics.evidence_count ?? diag.evidence_count)
    push('tokens', jobOverview?.usage?.total_tokens)
    push('notes', diag.notes || diag.summary)
    return out
  }, [diagnostic, jobOverview])
  const [pollingFallback, setPollingFallback] = useState(false)
  const [renameLoadingId, setRenameLoadingId] = useState(null)
  const [deleteLoadingId, setDeleteLoadingId] = useState(null)
  const [retryLoadingId, setRetryLoadingId] = useState(null)
  const [toasts, setToasts] = useState([])
  const bottomRef = useRef(null)
  const sseRef = useRef(null)
  const retryRef = useRef(0)

  const addToast = (message, type = 'info') => {
    const id = Date.now() + Math.random()
    setToasts((prev) => [...prev, { id, message, type }])
    setTimeout(() => {
      setToasts((prev) => prev.filter((t) => t.id !== id))
    }, 4000)
  }

  useEffect(() => {
    initSentry()
    loadDashboard()
    const saved = localStorage.getItem('activeJobId')
    loadJobs()
    if (saved) {
      const savedId = parseInt(saved, 10)
      if (!Number.isNaN(savedId)) {
        handleSelectJob({ id: savedId, name: `job-${savedId}` })
      }
    }
  }, [])

  useEffect(() => {
    if (!showRenameDialog) {
      setRenameTarget(null)
      setRenameValue('')
    }
  }, [showRenameDialog])

  useEffect(() => {
    if (!showDeleteDialog) {
      setDeleteTarget(null)
      setDeleteLoadingId(null)
    }
  }, [showDeleteDialog])

  useEffect(() => {
    if (!showRetryDialog) {
      setRetryTarget(null)
      setRetryIterations(5)
      setRetryDuration(300)
      setRetryBudget(0)
      setRetryLoadingId(null)
    }
  }, [showRetryDialog])

  useEffect(() => {
    if (!showPromptDialog) return
    api
      .getSystemPrompt()
      .then((res) => setPromptText(res.system_prompt || ''))
      .catch((e) => console.warn('system prompt load', e))
    api
      .getPromptVariant()
      .then((res) => setPromptVariant(res.variant || ''))
      .catch((e) => console.warn('variant load', e))
  }, [showPromptDialog])

  useEffect(() => {
    if (bottomRef.current) bottomRef.current.scrollIntoView({ behavior: 'smooth' })
  }, [messages, timeline])

  // Auto-clear notice after a short delay to avoid stale banners
  useEffect(() => {
    if (!notice) return
    const t = setTimeout(() => setNotice(''), 4000)
    return () => clearTimeout(t)
  }, [notice])

  useEffect(() => {
    if (!activeJob?.job_id) return
    const id = activeJob.job_id
    const interval = setInterval(async () => {
      try {
        const st = await api.jobStatus(id)
        setJobStatus(st)
        const tl = await api.jobTimeline(id)
        setTimeline(tl.timeline || [])
        const ev = await api.jobEvidence(id, 100)
        setEvidence(ev.evidence || [])
      } catch (e) {
        console.warn(e)
      }
    }, 8000)
    return () => clearInterval(interval)
  }, [activeJob])

  useEffect(() => {
    if (!activeJob?.job_id) return
    const id = activeJob.job_id
    retryRef.current = 0
    if (sseRef.current) {
      sseRef.current.close()
      sseRef.current = null
    }
    const es = new EventSource(`/api/jobs/${id}/timeline/stream`)
    sseRef.current = es
    es.onmessage = (evt) => {
      try {
        const data = JSON.parse(evt.data)
        setJobStatus((prev) => ({ ...(prev || {}), status: data.status }))
        if (data.timeline) setTimeline(data.timeline)
        if (data.timeline && data.timeline.length) {
          api.jobEvidence(id, 100).then((ev) => setEvidence(ev.evidence || [])).catch(() => {})
          const tools = []
          const msgs = []
          ;(data.timeline || []).forEach((entry) => {
            ;(entry.messages || []).forEach((m) => {
              if (m.tool || m.tool_call) {
                tools.push({
                  tool: m.tool || m.tool_call,
                  agent: m.agent || m.role,
                  content: m.content || m.hypothesis || '',
                  meta: m.meta,
                })
              } else {
                msgs.push({
                  role: m.role || m.agent || 'agent',
                  content: m.content || m.hypothesis || '',
                  meta: `${m.agent || m.role || 'agent'} Â· it${entry.iteration || '?'}`
                })
              }
            })
          })
          setToolOutputs(tools)
          setMessages(msgs)
          api.jobOverview(id).then((ov) => setJobOverview(ov)).catch(() => {})
          api.jobDashboard(id).then((diag) => setDiagnostic(diag)).catch(() => {})
        }
        retryRef.current = 0
        setSseError('')
        setPollingFallback(false)
        logClientEvent('sse_message', { job_id: id, status: data.status, events: data.timeline?.length })
      } catch (e) {
        console.warn('SSE parse', e)
        setSseError('Erreur de flux SSE (parse), bascule sur polling...')
        setPollingFallback(true)
        logClientEvent('sse_parse_error', { job_id: id, error: String(e) })
        captureClientError(e, { job_id: id, stage: 'sse_parse' })
      }
    }
    es.onerror = () => {
      console.error('SSE error, switching to polling')
      es.close()
      retryRef.current += 1
      if (retryRef.current < 5) {
        setTimeout(() => setActiveJob((prev) => ({ ...prev })), 1500 * retryRef.current)
      }
      setSseError('Flux SSE interrompu, tentative de reconnexion...')
      setPollingFallback(true)
      addToast('Flux SSE interrompu: mode polling actif', 'error')
      logClientEvent('sse_error', { job_id: id, attempt: retryRef.current })
        captureClientError(new Error('sse_error'), { job_id: id, attempt: retryRef.current })
    }
    return () => {
      es.close()
      sseRef.current = null
    }
  }, [activeJob?.job_id])

  const loadJobs = async () => {
    try {
      const res = await api.listJobs()
      setJobsHistory(res.jobs || [])
      // auto-select latest job if none active
      if (!activeJob && res.jobs && res.jobs.length) {
        handleSelectJob(res.jobs[0])
      }
    } catch (e) {
      console.warn('jobs history', e)
    }
  }

  const loadDashboard = async () => {
    setDashboardLoading(true)
    try {
      const res = await api.dashboard()
      setDashboard(res)
    } catch (e) {
      console.warn('dashboard', e)
    } finally {
      setDashboardLoading(false)
    }
  }

  const handleSend = async (e) => {
    e.preventDefault()
    if (!input.trim()) return
    setError('')
    setSseError('')
    const userMsg = { role: 'user', content: input.trim(), meta: 'Vous' }
    setMessages((prev) => [...prev, userMsg])
    setInput('')
    try {
      const job = await api.startJob(`chat-${Date.now()}`, input.trim())
      setActiveJob(job)
      setJobStatus({ status: 'started' })
      setTimeline([])
      setEvidence([])
      addToast('Job demarre', 'success')
    } catch (err) {
      setError(err.message)
      addToast(err.message, 'error')
    }
  }

  const handleSelectJob = async (job) => {
    localStorage.setItem('activeJobId', job.id)
    setActiveJob({ job_id: job.id, name: job.name })
    setSseError('')
    try {
      const st = await api.jobStatus(job.id)
      setJobStatus(st)
      const tl = await api.jobTimeline(job.id)
      setTimeline(tl.timeline || [])
      // Flatten tool outputs and messages from timeline for replay
      const tools = []
      const msgs = []
      ;(tl.timeline || []).forEach((entry) => {
        ;(entry.messages || []).forEach((m) => {
          if (m.tool) {
            tools.push({ tool: m.tool, agent: m.agent || m.role, content: m.content, meta: m.meta })
          } else {
            msgs.push({
              role: m.role || m.agent || 'agent',
              content: m.content || m.hypothesis || '',
              meta: `${m.agent || m.role || 'agent'} Â· it${entry.iteration || '?'}`
            })
          }
        })
      })
      setToolOutputs(tools)
      setMessages(msgs)
      const ov = await api.jobOverview(job.id)
      setJobOverview(ov)
      const diag = await api.jobDashboard(job.id)
      setDiagnostic(diag)
      const ev = await api.jobEvidence(job.id)
      setEvidence(ev.evidence || [])
      setShowHistory(false)
    } catch (e) {
      setError(e.message)
    }
  }

  const handleNewConversation = () => {
    setActiveJob(null)
    setJobStatus(null)
    setTimeline([])
    setEvidence([])
    setMessages([])
    setError('')
    setInput('')
    setSseError('')
    localStorage.removeItem('activeJobId')
  }

  const handleCreateJob = async (e) => {
    e?.preventDefault()
    const query = newQuery.trim()
    if (!query) {
      setError('La requete est requise')
      return
    }
    const autoName = query.slice(0, 60) || `chat-${Date.now()}`
    setError('')
    setSseError('')
    try {
      const job = await api.startJob(autoName, query)
      setActiveJob(job)
      setJobStatus({ status: 'started' })
      setTimeline([])
      setEvidence([])
      setMessages([{ role: 'user', content: query, meta: 'Vous' }])
      setShowNewDialog(false)
      setNewQuery('')
      loadJobs()
      addToast('Conversation demarree', 'success')
    } catch (err) {
      setError(err.message)
      addToast(err.message, 'error')
    }
  }

  const handleConfirmRename = async () => {
    if (!renameTarget) return
    const value = renameValue.trim()
    if (!value) {
      setError('Le nom est requis')
      return
    }
    try {
      setRenameLoadingId(renameTarget.id)
      await api.renameJob(renameTarget.id, value)
      if (activeJob?.job_id === renameTarget.id) {
        setActiveJob((prev) => ({ ...(prev || {}), name: value }))
      }
      await loadJobs()
      setShowRenameDialog(false)
      addToast('Conversation renommee', 'success')
    } catch (err) {
      setError(err.message)
      addToast(err.message, 'error')
    } finally {
      setRenameLoadingId(null)
    }
  }

  const handleConfirmDelete = async () => {
    if (!deleteTarget) return
    try {
      setDeleteLoadingId(deleteTarget.id)
      await api.deleteJob(deleteTarget.id)
      await loadJobs()
      if (activeJob?.job_id === deleteTarget.id) handleNewConversation()
      setShowDeleteDialog(false)
      addToast('Conversation supprimee', 'success')
    } catch (err) {
      setError(err.message)
      addToast(err.message, 'error')
    } finally {
      setDeleteLoadingId(null)
    }
  }

  const handleConfirmRetry = async () => {
    if (!retryTarget) return
    try {
      setRetryLoadingId(retryTarget.id)
      const nj = await api.retryJob(retryTarget.id, {
        max_iterations: retryIterations,
        max_duration_seconds: retryDuration,
        max_token_budget: retryBudget,
      })
      await loadJobs()
      setActiveJob({ job_id: nj.job_id, name: `retry-${retryTarget.name || retryTarget.id}` })
      setTimeline([])
      setEvidence([])
      setMessages([])
      setJobStatus({ status: 'started' })
      setShowRetryDialog(false)
      addToast('Relance en cours', 'success')
    } catch (err) {
      setError(err.message)
      addToast(err.message, 'error')
    } finally {
      setRetryLoadingId(null)
    }
  }

  const handleVectorSearch = async (e) => {
    e?.preventDefault()
    const q = vectorQuery.trim()
    if (!q) return
    setVectorLoading(true)
    setError('')
    try {
      const res = await api.searchVector(q, 12)
      const seen = new Set()
      const deduped = []
      ;(res.results || []).forEach((r) => {
        const key = r.metadata?.source || r.metadata?.document_id || r.metadata?.text || r.id || r.score
        if (seen.has(key)) return
        seen.add(key)
        deduped.push(r)
      })
      setVectorResults(deduped)
      addToast(`Recherche vectorielle: ${deduped.length} resultats`, 'info')
    } catch (err) {
      setError(err.message)
      addToast(err.message, 'error')
    } finally {
      setVectorLoading(false)
    }
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-950 via-slate-900 to-slate-950 text-foreground">
      <div className="flex h-screen">
        {/* Side menu with collapse */}
        <aside className={`${sidebarOpen ? 'w-64' : 'w-0 md:w-12'} transition-all duration-200 overflow-hidden border-r border-border bg-background/60 backdrop-blur flex-col hidden md:flex`}>
          <div className="p-4 border-b border-border flex items-center justify-between">
            <h2 className="font-semibold">Historique</h2>
            <div className="flex gap-2 items-center">
              <button className="btn-ghost text-xs" onClick={() => setSidebarOpen((v) => !v)}>{sidebarOpen ? 'Fermer' : 'Ouvrir'}</button>
              <button className="btn-secondary text-sm" onClick={loadJobs}>Rafraichir</button>
            </div>
          </div>
          <div className="flex-1 overflow-auto p-3 space-y-2">
            {jobsHistory.map((j) => (
              <div
                key={j.id}
                className="rounded-xl border border-border bg-card p-3 text-sm space-y-1 cursor-pointer hover:border-primary"
                onClick={() => handleSelectJob(j)}
              >
                <div className="flex items-center justify-between">
                  <span className="font-semibold">{j.name}</span>
                  <span className="text-xs text-muted-foreground">{j.status}</span>
                </div>
                <div className="text-xs text-muted-foreground">{new Date(j.created_at).toLocaleString()}</div>
                <div className="flex gap-2">
                  <button
                    type="button"
                    className="btn-ghost text-xs"
                    disabled={renameLoadingId === j.id}
                    onClick={(e) => {
                      e.stopPropagation()
                      setRenameTarget(j)
                      setRenameValue(j.name || '')
                      setShowRenameDialog(true)
                    }}
                  >
                    {renameLoadingId === j.id ? '...' : 'Renommer'}
                  </button>
                  <button
                    type="button"
                    className="btn-ghost text-xs text-red-400"
                    disabled={deleteLoadingId === j.id}
                    onClick={(e) => {
                      e.stopPropagation()
                      setDeleteTarget(j)
                      setShowDeleteDialog(true)
                    }}
                  >
                    {deleteLoadingId === j.id ? '...' : 'Supprimer'}
                  </button>
                  <button
                    type="button"
                    className="btn-ghost text-xs text-blue-400"
                    disabled={retryLoadingId === j.id}
                    onClick={(e) => {
                      e.stopPropagation()
                      setRetryTarget(j)
                      setRetryIterations(5)
                      setShowRetryDialog(true)
                    }}
                  >
                    {retryLoadingId === j.id ? '...' : 'Relancer'}
                  </button>
                </div>
              </div>
            ))}
          </div>
          <div className="p-4 border-t border-border">
            <button
              className="btn-primary w-full"
              onClick={() => {
                handleNewConversation()
                setShowNewDialog(true)
              }}
            >
              + Nouvelle conversation
            </button>
          </div>
        </aside>

        <div className="flex-1 flex flex-col">
          <header className="flex items-center justify-between border-b border-border px-4 py-3">
            <div className="flex items-center gap-3">
              <button className="md:hidden btn-ghost" onClick={() => setShowHistory(true)}>Menu</button>
              <button className="hidden md:inline-flex btn-ghost" onClick={() => setSidebarOpen((v) => !v)}>{sidebarOpen ? 'â†”' : 'â˜°'}</button>
              <div>
                <p className="text-xs text-muted-foreground">SearchOne</p>
                <h1 className="text-lg font-semibold">Chat & Recherche</h1>
              </div>
            </div>
            <div className="flex items-center gap-3 text-sm text-muted-foreground">
              <span>Job: {activeJob?.job_id || 'aucun'}</span>
              <span className="rounded-full bg-muted px-3 py-1 text-xs flex items-center gap-2">
                {jobStatus?.status || 'idle'}
                {pollingFallback && <span className="rounded-full bg-yellow-500/20 text-yellow-100 px-2 py-[2px]">polling</span>}
              </span>
              <button className="btn-secondary" onClick={() => setShowPromptDialog(true)}>Prompt</button>
            </div>
          </header>

          <main className="relative flex-1 overflow-auto px-4 py-3 space-y-4">
            <section className="grid gap-3 md:grid-cols-3">
              <div className="rounded-2xl border border-border bg-card p-3 text-sm">
                <div className="flex items-center justify-between">
                  <span className="text-xs text-muted-foreground">Documents</span>
                  <button className="btn-ghost text-xs" onClick={() => { loadDashboard(); loadJobs(); }}>
                    {dashboardLoading ? '...' : 'RafraÃ®chir'}
                  </button>
                </div>
                <div className="text-2xl font-semibold">{dashboard?.db?.documents ?? '-'}</div>
                <div className="text-xs text-muted-foreground">Chunks: {dashboard?.db?.chunks ?? '-'}</div>
              </div>
              <div className="rounded-2xl border border-border bg-card p-3 text-sm">
                <div className="flex items-center justify-between">
                  <span className="text-xs text-muted-foreground">Jobs</span>
                  <button
                    className="btn-ghost text-xs"
                    onClick={async () => {
                      try {
                        await api.consolidateKnowledge()
                        addToast('Connaissance consolidÃ©e', 'success')
                      } catch (e) {
                        addToast('Consolidation Ã©chouÃ©e', 'error')
                      }
                    }}
                  >
                    Consolider
                  </button>
                </div>
                <div className="text-2xl font-semibold">{dashboard?.db?.jobs ?? '-'}</div>
                <div className="text-xs text-muted-foreground">Schedules: {dashboard?.schedules?.length ?? 0}</div>
                <div className="text-[11px] text-muted-foreground mt-1">
                  Dernier job: {jobOverview?.status || '-'} | timeline: {jobOverview?.timeline_len ?? 0}
                </div>
              </div>
              <div className="rounded-2xl border border-border bg-card p-3 text-sm">
                <div className="flex items-center justify-between">
                  <span className="text-xs text-muted-foreground">Dernier diagnostic</span>
                  <span className="text-[11px] text-muted-foreground">{dashboard?.latest_job_diagnostic?.status}</span>
                </div>
                <div className="flex flex-wrap gap-2">
                  {diagnosticItems.map((d, idx) => (
                    <span key={idx} className="rounded-full bg-muted px-3 py-1 text-[11px] text-foreground border border-border/60">
                      {d.label}: <span className="font-semibold">{String(d.value).slice(0, 60)}</span>
                    </span>
                  ))}
                  {!diagnosticItems.length && <span className="text-xs text-muted-foreground">n/a</span>}
                </div>
              </div>
              {jobOverview && (
                <>
                  <div className="rounded-2xl border border-border bg-card p-3 text-sm">
                    <div className="text-xs text-muted-foreground">Coverage</div>
                    <div className="text-2xl font-semibold">{(jobOverview.run_metrics?.coverage_score ?? 0).toFixed?.(2) ?? '-'}</div>
                  </div>
                  <div className="rounded-2xl border border-border bg-card p-3 text-sm">
                    <div className="text-xs text-muted-foreground">Evidence</div>
                    <div className="text-2xl font-semibold">{jobOverview.run_metrics?.evidence_count ?? '-'}</div>
                  </div>
                  <div className="rounded-2xl border border-border bg-card p-3 text-sm">
                    <div className="text-xs text-muted-foreground">Tokens</div>
                    <div className="text-2xl font-semibold">{jobOverview.usage?.total_tokens ?? '-'}</div>
                  </div>
                </>
              )}
            </section>
            {sseError && (
              <div className="rounded-xl border border-yellow-500/60 bg-yellow-500/10 text-yellow-200 px-3 py-2 text-sm">
                {sseError}
                {pollingFallback && <span className="ml-2 text-xs text-emerald-200">(mode polling)</span>}
              </div>
            )}
            {/* Toasts overlay (top-right) */}
            <div className="fixed top-4 right-4 space-y-2 z-10">
              {toasts.map((t) => (
                <div
                  key={t.id}
                  className={`rounded-xl border px-3 py-2 text-sm shadow-md bg-background/90 backdrop-blur ${
                    t.type === 'error'
                      ? 'border-red-500/60 text-red-100'
                      : t.type === 'success'
                        ? 'border-emerald-500/60 text-emerald-100'
                        : 'border-blue-500/60 text-blue-100'
                  }`}
                >
                  {t.message}
                </div>
              ))}
            </div>
            <section className="space-y-3">
              {messages.map((m, idx) => (
                <MessageBubble key={idx} role={m.role} content={m.content} meta={m.meta} />
              ))}
              {timeline.map((t, idx) => (
                <TimelineCard key={`tl-${idx}`} entry={t} />
              ))}
              <div ref={bottomRef} />
            </section>

            <section className="grid gap-3 md:grid-cols-2">
              <div className="space-y-2">
                <div className="flex items-center justify-between">
                  <h2 className="text-sm font-semibold text-muted-foreground">Preuves / rÃ©sultats</h2>
                  {activeJob?.job_id && (
                    <button
                      className="btn-secondary"
                      onClick={async () => {
                        try {
                          const ev = await api.jobEvidence(activeJob.job_id, 100)
                          setEvidence(ev.evidence || [])
                          const ov = await api.jobOverview(activeJob.job_id)
                          setJobOverview(ov)
                          const diag = await api.jobDashboard(activeJob.job_id)
                          setDiagnostic(diag)
                        } catch (e) {
                          setError(e.message)
                        }
                      }}
                    >
                      RafraÃ®chir
                    </button>
                  )}
                </div>
                <div className="grid gap-3">
                  {evidence.map((ev, idx) => (
                    <div key={idx} className="rounded-xl border border-border bg-card p-3 text-sm space-y-2">
                      <div className="flex items-center justify-between text-xs text-muted-foreground">
                        <span>Doc {ev.document_id ?? '-'}</span>
                        <span>Score {ev.score?.toFixed?.(3) ?? '-'}</span>
                      </div>
                      <div className="text-xs text-muted-foreground">{ev.source || ev.meta?.source}</div>
                      <div className="text-foreground whitespace-pre-wrap">{(ev.text || '').slice(0, 300)}</div>
                    </div>
                  ))}
                  {!evidence.length && <p className="text-sm text-muted-foreground">Aucune preuve pour le moment.</p>}
                </div>
              </div>

              <div className="space-y-2">
                <div className="flex items-center justify-between">
                  <h2 className="text-sm font-semibold text-muted-foreground">Outils / exÃ©cutions</h2>
                  <span className="text-xs text-muted-foreground">{toolOutputs.length} appels</span>
                </div>
                <div className="grid gap-2">
                  {toolOutputs.map((t, idx) => (
                    <div key={idx} className="rounded-xl border border-border bg-card p-3 text-sm space-y-1">
                      <div className="flex items-center justify-between text-xs text-muted-foreground">
                        <span>{t.agent || 'agent'} â€¢ {t.tool}</span>
                      </div>
                      <div className="text-foreground whitespace-pre-wrap text-sm">{t.content}</div>
                    </div>
                  ))}
                  {!toolOutputs.length && <p className="text-sm text-muted-foreground">Aucun appel d'outil pour le moment.</p>}
                </div>
              </div>
            </section>

            <section className="space-y-2">
              <div className="flex items-center justify-between">
                <h2 className="text-sm font-semibold text-muted-foreground">Recherche vectorielle</h2>
                <button
                  className="btn-secondary"
                  onClick={() => {
                    setVectorQuery('')
                    setVectorResults([])
                  }}
                >
                  Effacer
                </button>
              </div>
              <form className="flex gap-2" onSubmit={handleVectorSearch}>
                <input
                  className="flex-1 rounded-xl border border-border bg-card px-3 py-2 text-sm text-foreground"
                  aria-label="RequÃªte vectorielle"
                  value={vectorQuery}
                  onChange={(e) => setVectorQuery(e.target.value)}
                />
                <button type="submit" className="btn-primary" disabled={vectorLoading}>
                  {vectorLoading ? '...' : 'Chercher'}
                </button>
              </form>
              <div className="grid gap-3 md:grid-cols-2">
                {vectorResults.map((r, idx) => (
                  <div key={idx} className="rounded-xl border border-border bg-card p-3 text-sm space-y-2">
                    <div className="flex items-center justify-between text-xs text-muted-foreground">
                      <span>{r.metadata?.source || 'Source inconnue'}</span>
                      <span>Score {r.score?.toFixed?.(3) ?? '-'}</span>
                    </div>
                    {r.metadata?.timestamp && (
                      <div className="text-[11px] text-muted-foreground">
                        {new Date(r.metadata.timestamp).toLocaleString()}
                      </div>
                    )}
                    <div className="text-foreground whitespace-pre-wrap">
                      {(r.metadata?.text || r.text || '').slice(0, 320) || '[aucun extrait]'}
                    </div>
                  </div>
                ))}
                {!vectorResults.length && (
                  <p className="text-sm text-muted-foreground">Aucun resultat vectoriel pour le moment.</p>
                )}
              </div>
            </section>
          </main>

          <footer className="border-t border-border bg-background/80 backdrop-blur px-4 py-3">
            {error && <div className="mb-2 text-sm text-red-400">{error}</div>}
            <form onSubmit={handleSend} className="flex items-center gap-2">
              <input
                value={input}
                onChange={(e) => setInput(e.target.value)}
                aria-label="Question ou requÃªte"
                className="flex-1 rounded-2xl border border-border bg-card px-4 py-3 text-sm focus:outline-none focus:ring-2 focus:ring-primary"
              />
              <button type="submit" className="btn-primary">
                Envoyer
              </button>
            </form>
          </footer>
        </div>
      </div>

      <Dialog.Root open={showHistory} onOpenChange={setShowHistory}>
        <Dialog.Portal>
          <Dialog.Overlay className="fixed inset-0 bg-black/60 backdrop-blur" />
          <Dialog.Content className="fixed top-0 left-0 h-full w-72 bg-background border-r border-border p-4 space-y-3">
            <div className="flex items-center justify-between">
              <Dialog.Title className="font-semibold">Historique</Dialog.Title>
              <Dialog.Close className="btn-ghost">Fermer</Dialog.Close>
            </div>
            <div className="space-y-2 overflow-auto h-[80vh]">
              {jobsHistory.map((j) => (
                <div
                  key={j.id}
                  className="rounded-xl border border-border bg-card p-3 text-sm space-y-1 cursor-pointer hover:border-primary"
                  onClick={() => handleSelectJob(j)}
                >
                  <div className="flex items-center justify-between">
                    <span className="font-semibold">{j.name}</span>
                    <span className="text-xs text-muted-foreground">{j.status}</span>
                  </div>
                  <div className="text-xs text-muted-foreground">{new Date(j.created_at).toLocaleString()}</div>
                  <div className="flex gap-2">
                    <button
                    type="button"
                    className="btn-ghost text-xs"
                    disabled={renameLoadingId === j.id}
                    onClick={(e) => {
                      e.stopPropagation()
                      setRenameTarget(j)
                      setRenameValue(j.name || '')
                      setShowRenameDialog(true)
                    }}
                  >
                    {renameLoadingId === j.id ? '...' : 'Renommer'}
                  </button>
                  <button
                    type="button"
                    className="btn-ghost text-xs text-red-400"
                    disabled={deleteLoadingId === j.id}
                    onClick={(e) => {
                      e.stopPropagation()
                      setDeleteTarget(j)
                      setShowDeleteDialog(true)
                    }}
                  >
                    {deleteLoadingId === j.id ? '...' : 'Supprimer'}
                  </button>
                  <button
                    type="button"
                    className="btn-ghost text-xs text-blue-400"
                    disabled={retryLoadingId === j.id}
                    onClick={(e) => {
                      e.stopPropagation()
                      setRetryTarget(j)
                      setRetryIterations(5)
                      setShowRetryDialog(true)
                    }}
                  >
                    {retryLoadingId === j.id ? '...' : 'Relancer'}
                  </button>
                  </div>
                </div>
              ))}
            </div>
            <button
              className="btn-primary w-full"
              onClick={() => {
                handleNewConversation()
                setShowHistory(false)
                setShowNewDialog(true)
              }}
            >
              + Nouvelle conversation
            </button>
          </Dialog.Content>
        </Dialog.Portal>
      </Dialog.Root>

      <Dialog.Root open={showRenameDialog} onOpenChange={setShowRenameDialog}>
        <Dialog.Portal>
          <Dialog.Overlay className="fixed inset-0 bg-black/60 backdrop-blur" />
          <Dialog.Content className="fixed top-1/2 left-1/2 w-[420px] max-w-[90vw] -translate-x-1/2 -translate-y-1/2 rounded-2xl border border-border bg-background p-4 space-y-3 shadow-lg">
            <Dialog.Title className="font-semibold">Renommer la conversation</Dialog.Title>
            <p className="text-sm text-muted-foreground">
              {renameTarget ? `Job #${renameTarget.id}` : 'Selectionnez une conversation a renommer.'}
            </p>
            <input
              className="w-full rounded-xl border border-border bg-card px-3 py-2 text-sm text-foreground"
              aria-label="Nouveau nom"
              value={renameValue}
              onChange={(e) => setRenameValue(e.target.value)}
            />
            <div className="flex justify-end gap-2">
              <Dialog.Close className="btn-ghost" onClick={() => setRenameTarget(null)}>
                Annuler
              </Dialog.Close>
              <button className="btn-primary" onClick={handleConfirmRename} disabled={!renameTarget}>
                Renommer
              </button>
            </div>
          </Dialog.Content>
        </Dialog.Portal>
      </Dialog.Root>

      <Dialog.Root open={showDeleteDialog} onOpenChange={setShowDeleteDialog}>
        <Dialog.Portal>
          <Dialog.Overlay className="fixed inset-0 bg-black/60 backdrop-blur" />
          <Dialog.Content className="fixed top-1/2 left-1/2 w-[420px] max-w-[90vw] -translate-x-1/2 -translate-y-1/2 rounded-2xl border border-border bg-background p-4 space-y-3 shadow-lg">
            <Dialog.Title className="font-semibold">Supprimer la conversation</Dialog.Title>
            <p className="text-sm text-muted-foreground">
              {deleteTarget
                ? `Confirmer la suppression de "${deleteTarget.name}" (job #${deleteTarget.id})`
                : 'Selectionnez une conversation a supprimer.'}
            </p>
            <div className="flex justify-end gap-2">
              <Dialog.Close className="btn-ghost" onClick={() => setDeleteTarget(null)}>
                Annuler
              </Dialog.Close>
              <button
                className="btn-primary bg-red-500 text-white hover:bg-red-600"
                onClick={handleConfirmDelete}
                disabled={!deleteTarget}
              >
                Supprimer
              </button>
            </div>
          </Dialog.Content>
        </Dialog.Portal>
      </Dialog.Root>

      <Dialog.Root open={showRetryDialog} onOpenChange={setShowRetryDialog}>
        <Dialog.Portal>
          <Dialog.Overlay className="fixed inset-0 bg-black/60 backdrop-blur" />
          <Dialog.Content className="fixed top-1/2 left-1/2 w-[420px] max-w-[90vw] -translate-x-1/2 -translate-y-1/2 rounded-2xl border border-border bg-background p-4 space-y-3 shadow-lg">
            <Dialog.Title className="font-semibold">Relancer le job</Dialog.Title>
            <p className="text-sm text-muted-foreground">
              {retryTarget ? `Relancer "${retryTarget.name}" (job #${retryTarget.id})` : 'Selectionnez un job.'}
            </p>
            <div className="space-y-2">
              <label className="text-xs text-muted-foreground">Iterations max</label>
              <input
                type="number"
                min="1"
                max="20"
                className="w-full rounded-xl border border-border bg-card px-3 py-2 text-sm text-foreground"
                value={retryIterations}
                onChange={(e) => setRetryIterations(parseInt(e.target.value, 10) || 1)}
              />
              <label className="text-xs text-muted-foreground">Duree max (secondes)</label>
              <input
                type="number"
                min="30"
                max="1200"
                className="w-full rounded-xl border border-border bg-card px-3 py-2 text-sm text-foreground"
                value={retryDuration}
                onChange={(e) => setRetryDuration(parseInt(e.target.value, 10) || 300)}
              />
              <label className="text-xs text-muted-foreground">Budget token max (0 = illimite)</label>
              <input
                type="number"
                min="0"
                max="20000"
                className="w-full rounded-xl border border-border bg-card px-3 py-2 text-sm text-foreground"
                value={retryBudget}
                onChange={(e) => setRetryBudget(parseInt(e.target.value, 10) || 0)}
              />
            </div>
            <div className="flex justify-end gap-2">
              <Dialog.Close className="btn-ghost" onClick={() => setRetryTarget(null)}>
                Annuler
              </Dialog.Close>
              <button className="btn-primary" onClick={handleConfirmRetry} disabled={!retryTarget || retryLoadingId}>
                {retryLoadingId ? '...' : 'Relancer'}
              </button>
            </div>
          </Dialog.Content>
        </Dialog.Portal>
      </Dialog.Root>

      <Dialog.Root open={showPromptDialog} onOpenChange={setShowPromptDialog}>
        <Dialog.Portal>
          <Dialog.Overlay className="fixed inset-0 bg-black/60 backdrop-blur" />
          <Dialog.Content className="fixed top-1/2 left-1/2 w-[420px] max-w-[90vw] -translate-x-1/2 -translate-y-1/2 rounded-2xl border border-border bg-background p-4 space-y-3 shadow-lg">
            <Dialog.Title className="font-semibold">Prompt global</Dialog.Title>
            <textarea
              className="w-full rounded-xl border border-border bg-card p-3 text-sm text-foreground min-h-[160px]"
              aria-label="System prompt global"
              value={promptText}
              onChange={(e) => setPromptText(e.target.value)}
            />
            <div className="flex items-center gap-2">
              <input
                className="w-full rounded-xl border border-border bg-card px-3 py-2 text-sm text-foreground"
                aria-label="Variant de prompt"
                value={promptVariant}
                onChange={(e) => setPromptVariant(e.target.value)}
              />
              <button
                className="btn-secondary whitespace-nowrap"
                type="button"
                onClick={async () => {
                  try {
                    await api.setPromptVariant(promptVariant)
                  } catch (e) {
                    setError('Impossible de definir la variante')
                  }
                }}
              >
                Variante
              </button>
            </div>
            <div className="flex justify-end gap-2">
              <Dialog.Close className="btn-ghost">Annuler</Dialog.Close>
              <button
                className="btn-primary"
                onClick={async () => {
                  try {
                    await api.setSystemPrompt(promptText)
                    setShowPromptDialog(false)
                  } catch (e) {
                    setError('Impossible de sauvegarder le prompt')
                  }
                }}
              >
                Sauvegarder
              </button>
            </div>
          </Dialog.Content>
        </Dialog.Portal>
      </Dialog.Root>

      <Dialog.Root open={showNewDialog} onOpenChange={setShowNewDialog}>
        <Dialog.Portal>
          <Dialog.Overlay className="fixed inset-0 bg-black/60 backdrop-blur" />
          <Dialog.Content className="fixed top-1/2 left-1/2 w-[420px] max-w-[90vw] -translate-x-1/2 -translate-y-1/2 rounded-2xl border border-border bg-background p-4 space-y-3 shadow-lg">
            <Dialog.Title className="font-semibold">Nouvelle recherche</Dialog.Title>
            <form className="space-y-3" onSubmit={handleCreateJob}>
              <textarea
                className="w-full rounded-xl border border-border bg-card p-3 text-sm text-foreground min-h-[120px]"
                aria-label="Requête"
                value={newQuery}
                onChange={(e) => setNewQuery(e.target.value)}
                placeholder='Décrivez la recherche à lancer'
              />
              <div className="flex justify-end gap-2">
                <Dialog.Close className="btn-ghost">Annuler</Dialog.Close>
                <button type="submit" className="btn-primary">
                  Chercher
                </button>
              </div>
            </form>
          </Dialog.Content>
        </Dialog.Portal>
      </Dialog.Root>
    </div>
  )
}






