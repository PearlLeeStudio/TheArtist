import { useEffect, useState } from 'react';

type Theme = 'dark' | 'light';

function getInitialTheme(): Theme {
  const stored = localStorage.getItem('theartist-theme');
  if (stored === 'light' || stored === 'dark') return stored;
  return 'dark';
}

export default function ThemeToggle() {
  const [theme, setTheme] = useState<Theme>(getInitialTheme);

  useEffect(() => {
    document.documentElement.classList.toggle('light', theme === 'light');
    localStorage.setItem('theartist-theme', theme);
  }, [theme]);

  return (
    <button
      onClick={() => setTheme(theme === 'dark' ? 'light' : 'dark')}
      className="inline-flex h-7 w-8 items-center justify-center text-base leading-none"
      style={{ background: 'var(--btn-bg)', color: 'var(--text-primary)' }}
      title={`Switch to ${theme === 'dark' ? 'light' : 'dark'} mode`}
      aria-label={`Switch to ${theme === 'dark' ? 'light' : 'dark'} mode`}
    >
      <span aria-hidden="true" style={{ fontSize: 8 }}>{theme === 'dark' ? '\uD83C\uDF19' : '\u2600\uFE0F'}</span>
    </button>
  );
}
