import { test, expect } from '@playwright/test'

const BACKEND = process.env.BACKEND_BASE_URL || ''

test.describe('chat UI flows', () => {
  test('start job from modal, see timeline + evidence (mocked backend)', async ({ page }) => {
    const jobs = []
    let id = 1

    // If BACKEND_BASE_URL set, let requests go through to that backend
    if (BACKEND) {
      // Real backend run: skip mocks and rely on live API
      await page.goto('/')
      await page.getByRole('button', { name: '+ Nouvelle conversation' }).click()
      await page.getByPlaceholder('Nom (optionnel)').fill('Job test')
      await page.getByPlaceholder(/Requete ou question/i).fill('Test query')
      await page.getByRole('button', { name: 'Demarrer' }).click()
      // Wait for timeline and evidence to show up (best effort)
      await expect(page.getByText(/Iteration|Mock step/i)).toBeVisible({ timeout: 30000 })
      return
    }

    await page.route('**/api/**', async (route) => {
      const url = new URL(route.request().url())
      const { pathname, searchParams } = url
      if (pathname === '/api/jobs' && route.request().method() === 'GET') {
        return route.fulfill({ json: { jobs } })
      }
      if (pathname === '/api/jobs/start') {
        const name = searchParams.get('name') || `job-${id}`
        const query = searchParams.get('query') || ''
        const job = {
          id,
          name,
          status: 'running',
          created_at: new Date().toISOString(),
          updated_at: new Date().toISOString(),
          query,
        }
        jobs.unshift(job)
        return route.fulfill({ json: { job_id: id++, status: 'started' } })
      }
      if (/^\/api\/jobs\/\d+$/.test(pathname)) {
        const jobId = parseInt(pathname.split('/').pop(), 10)
        const job = jobs.find((j) => j.id === jobId)
        return route.fulfill({ json: job || { id: jobId, status: 'running', name: 'job' } })
      }
      if (/^\/api\/jobs\/\d+\/retry$/.test(pathname)) {
        const originId = parseInt(pathname.split('/').pop(), 10)
        const origin = jobs.find((j) => j.id === originId)
        const newJob = {
          id,
          name: `retry-${origin?.name || originId}`,
          status: 'started',
          created_at: new Date().toISOString(),
          updated_at: new Date().toISOString(),
        }
        jobs.unshift(newJob)
        return route.fulfill({ json: { job_id: id++, status: 'started', origin_job: originId } })
      }
      if (/^\/api\/jobs\/\d+\/timeline$/.test(pathname)) {
        return route.fulfill({
          json: {
            timeline: [
              {
                iteration: 1,
                summary: 'Mock step',
                messages: [{ agent: 'AgentX', role: 'assistant', content: 'Hi from agent' }],
              },
            ],
          },
        })
      }
      if (/^\/api\/jobs\/\d+\/evidence/.test(pathname)) {
        return route.fulfill({
          json: { evidence: [{ document_id: 1, score: 0.9, text: 'Mock evidence text' }] },
        })
      }
      // default passthrough
      return route.fulfill({ json: {} })
    })

    // Instead of overriding EventSource globally, rely on the timeline fetch
    await page.goto('/')

    await page.getByRole('button', { name: '+ Nouvelle conversation' }).click()
    await page.getByPlaceholder('Nom (optionnel)').fill('Job test')
    await page.getByPlaceholder(/Requete ou question/i).fill('Test query')
    await page.getByRole('button', { name: 'Demarrer' }).click()

    await expect(page.getByText('Test query')).toBeVisible()
    await expect(page.getByText('Mock step')).toBeVisible()
    await expect(page.getByText('Mock evidence text')).toBeVisible()
  })

  test('rename and delete history items (mocked backend)', async ({ page }) => {
    const jobs = [
      {
        id: 1,
        name: 'Job A',
        status: 'completed',
        created_at: new Date().toISOString(),
        updated_at: new Date().toISOString(),
      },
    ]
    let nextId = 2

    await page.route('**/api/**', async (route) => {
      const url = new URL(route.request().url())
      const { pathname, searchParams } = url
      if (pathname === '/api/jobs' && route.request().method() === 'GET') {
        return route.fulfill({ json: { jobs } })
      }
      if (/^\/api\/jobs\/\d+\/rename$/.test(pathname)) {
        const jobId = parseInt(pathname.split('/')[3], 10)
        const newName = searchParams.get('new_name')
        const job = jobs.find((j) => j.id === jobId)
        if (job && newName) job.name = newName
        return route.fulfill({ json: { id: jobId, name: newName, status: 'renamed' } })
      }
      if (/^\/api\/jobs\/\d+$/.test(pathname) && route.request().method() === 'GET') {
        const jobId = parseInt(pathname.split('/').pop(), 10)
        const job = jobs.find((j) => j.id === jobId)
        return route.fulfill({ json: job || { id: jobId, status: 'completed', name: 'Job' } })
      }
      if (/^\/api\/jobs\/\d+\/retry$/.test(pathname)) {
        const originId = parseInt(pathname.split('/').pop(), 10)
        const origin = jobs.find((j) => j.id === originId)
        const newJob = {
          id: nextId++,
          name: `retry-${origin?.name || originId}`,
          status: 'started',
          created_at: new Date().toISOString(),
          updated_at: new Date().toISOString(),
        }
        jobs.unshift(newJob)
        return route.fulfill({ json: { job_id: newJob.id, status: 'started', origin_job: originId } })
      }
      if (/^\/api\/jobs\/\d+$/.test(pathname) && route.request().method() === 'DELETE') {
        const jobId = parseInt(pathname.split('/').pop(), 10)
        const idx = jobs.findIndex((j) => j.id === jobId)
        if (idx >= 0) jobs.splice(idx, 1)
        return route.fulfill({ json: { id: jobId, status: 'deleted' } })
      }
      if (/^\/api\/jobs\/\d+\/timeline/.test(pathname)) {
        return route.fulfill({ json: { timeline: [] } })
      }
      return route.fulfill({ json: {} })
    })

    page.on('dialog', (dialog) => {
      if (dialog.type() === 'prompt') {
        dialog.accept('Job Renamed')
      } else {
        dialog.accept()
      }
    })

    await page.goto('/')
    await expect(page.getByText('Job A')).toBeVisible()

    await page.getByRole('button', { name: 'Renommer' }).first().click()
    await expect(page.getByText('Job Renamed')).toBeVisible()

    await page.getByRole('button', { name: 'Relancer' }).first().click()
    await page.getByRole('button', { name: 'Relancer' }).last().click()
    await expect(page.getByText(/retry-Job Renamed/)).toBeVisible()
    await expect(page.getByText(/Relance en cours|Relance/)).toBeVisible()

    await page.getByRole('button', { name: 'Supprimer' }).first().click()
    await expect(page.getByText('Job Renamed')).toHaveCount(0)
  })

  test('SSE fallback to polling shows banner (mocked SSE error)', async ({ page }) => {
    const jobs = [{ id: 1, name: 'Job SSE', status: 'running', created_at: new Date().toISOString(), updated_at: new Date().toISOString() }]
    await page.route('**/api/**', async (route) => {
      const url = new URL(route.request().url())
      const { pathname, searchParams } = url
      if (pathname === '/api/jobs' && route.request().method() === 'GET') {
        return route.fulfill({ json: { jobs } })
      }
      if (pathname === '/api/jobs/start') {
        jobs.unshift({
          id: 2,
          name: searchParams.get('name'),
          status: 'running',
          created_at: new Date().toISOString(),
          updated_at: new Date().toISOString(),
        })
        return route.fulfill({ json: { job_id: 2, status: 'started' } })
      }
      if (/^\/api\/jobs\/\d+$/.test(pathname)) {
        const jobId = parseInt(pathname.split('/').pop(), 10)
        const job = jobs.find((j) => j.id === jobId)
        return route.fulfill({ json: job || { id: jobId, status: 'running', name: 'job' } })
      }
      if (/^\/api\/jobs\/\d+\/timeline$/.test(pathname)) {
        return route.fulfill({ json: { timeline: [] } })
      }
      if (/^\/api\/jobs\/\d+\/evidence/.test(pathname)) {
        return route.fulfill({ json: { evidence: [] } })
      }
      return route.fulfill({ json: {} })
    })

    // Mock EventSource to trigger error once
    await page.addInitScript(() => {
      const OriginalES = window.EventSource
      class FailingES {
        constructor(url) {
          this.url = url
          setTimeout(() => {
            if (this.onerror) this.onerror(new Event('error'))
          }, 50)
        }
        close() {}
      }
      // @ts-ignore
      window.EventSource = FailingES
      // restore after load
      window.__restoreES = () => {
        // @ts-ignore
        window.EventSource = OriginalES
      }
    })

    await page.goto('/')
    await page.getByRole('button', { name: '+ Nouvelle conversation' }).click()
    await page.getByPlaceholder(/Requete ou question/i).fill('Test SSE fallback')
    await page.getByRole('button', { name: 'Demarrer' }).click()
    // banner or toast indicating polling mode
    await expect(page.getByText(/polling|Flux SSE interrompu/i)).toBeVisible({ timeout: 5000 })
    // restore EventSource
    await page.evaluate(() => {
      // @ts-ignore
      if (window.__restoreES) window.__restoreES()
    })
  })
})
