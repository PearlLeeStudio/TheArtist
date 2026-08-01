import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import './index.css'
import App from './App.tsx'
import { installTimingConsole } from './engine/experienceTiming.ts'

installTimingConsole()

// `?reset=1` wipes everything the app keeps in localStorage (current song,
// saved sessions, Generate and Play settings, theme), drops the parameter
// from the address bar and reloads. The reload matters: the stores already
// read localStorage while their modules evaluated, before this runs. A
// plain reload keeps everything.
const params = new URLSearchParams(window.location.search)
if (params.get('reset') === '1') {
  try {
    localStorage.clear()
  } catch {
    // storage unavailable — nothing to clear
  }
  params.delete('reset')
  const query = params.toString()
  window.history.replaceState(
    window.history.state,
    '',
    `${window.location.pathname}${query ? `?${query}` : ''}${window.location.hash}`,
  )
  window.location.reload()
} else {
  createRoot(document.getElementById('root')!).render(
    <StrictMode>
      <App />
    </StrictMode>,
  )
}
