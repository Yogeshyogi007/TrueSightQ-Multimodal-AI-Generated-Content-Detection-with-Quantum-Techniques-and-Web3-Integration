import { useEffect, useState } from 'react'
import Head from 'next/head'
import '../styles/paper.css'

export default function App({ Component, pageProps }) {
  const [theme, setTheme] = useState('light')

  useEffect(() => {
    const stored = typeof window !== 'undefined' ? localStorage.getItem('ai-detector-theme') : null
    const initial = stored === 'dark' ? 'dark' : 'light'
    setTheme(initial)

    // Manage an extra <link> for dark theme overlay (globals.css)
    const ensureLink = (mode) => {
      const id = 'theme-link-dark'
      let link = document.getElementById(id)
      if (mode === 'dark') {
        if (!link) {
          link = document.createElement('link')
          link.id = id
          link.rel = 'stylesheet'
          link.href = '/globals.css'
          document.head.appendChild(link)
        }
      } else {
        if (link && link.parentNode) link.parentNode.removeChild(link)
      }
    }

    ensureLink(initial)
    document.documentElement.setAttribute('data-theme', initial)
  }, [])

  const toggleTheme = () => {
    const next = theme === 'light' ? 'dark' : 'light'
    setTheme(next)
    if (typeof window !== 'undefined') {
      localStorage.setItem('ai-detector-theme', next)
      document.documentElement.setAttribute('data-theme', next)
      const id = 'theme-link-dark'
      let link = document.getElementById(id)
      if (next === 'dark') {
        if (!link) {
          link = document.createElement('link')
          link.id = id
          link.rel = 'stylesheet'
          link.href = '/globals.css'
          document.head.appendChild(link)
        }
      } else {
        if (link && link.parentNode) link.parentNode.removeChild(link)
      }
    }
  }

  return (
    <>
      <Head>
        <meta name="theme-color" content={theme === 'dark' ? '#0a0a0a' : '#fdfdfd'} />
      </Head>
      <button
        onClick={toggleTheme}
        aria-label="Toggle theme"
        title={theme === 'dark' ? '' : 'just a fancy name for dark theme'}
        style={{
          position: 'fixed',
          bottom: 16,
          right: 16,
          zIndex: 10000,
          padding: '10px 12px',
          borderRadius: 9999,
          border: '1px solid rgba(0,0,0,0.08)',
          // Swapped colors: green for light (Matrix mode), blue for dark
          background: theme === 'dark'
            ? 'linear-gradient(135deg,#1565C0,#00838F)'  // dark theme active → blue button (to go back to light)
            : 'linear-gradient(135deg,#00ff41,#00cccc)', // light theme active → green Matrix button
          color: '#fff',
          boxShadow: '0 6px 18px rgba(0,0,0,0.25)',
          cursor: 'pointer'
        }}
      >
        {theme === 'dark' ? 'Light Theme' : 'Matrix Mode'}
      </button>
      <Component {...pageProps} />
    </>
  )
}