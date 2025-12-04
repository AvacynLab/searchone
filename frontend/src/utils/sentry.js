import { logClientEvent } from './logger'

// Optional Sentry capture; toggled by global SEARCHONE_SENTRY_DSN (injected via HTML/env)
export const SENTRY_DSN =
  (typeof window !== 'undefined' && window.SEARCHONE_SENTRY_DSN) || process.env.SEARCHONE_SENTRY_DSN || null

let sentryClient = null

export const initSentry = async () => {
  if (!SENTRY_DSN || sentryClient) return
  try {
    const Sentry = await import('@sentry/browser')
    Sentry.init({ dsn: SENTRY_DSN, tracesSampleRate: 0.1, integrations: [] })
    sentryClient = Sentry
    logClientEvent('sentry_init', { ok: true })
  } catch (e) {
    logClientEvent('sentry_init_failed', { error: String(e) })
  }
}

export const captureClientError = (error, context = {}) => {
  if (!sentryClient) return
  try {
    sentryClient.captureException(error, { extra: context })
  } catch (e) {
    /* noop */
  }
}
