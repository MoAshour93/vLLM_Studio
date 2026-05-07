import { useCallback, useEffect } from 'react';
import { useAppStore } from '../store/index';

export function useTheme() {
  const theme = useAppStore((s) => s.theme);
  const setTheme = useAppStore((s) => s.setTheme);

  useEffect(() => {
    const stored = localStorage.getItem('vllm-studio-theme');
    if (stored === 'dark' || stored === 'light' || stored === 'system') {
      setTheme(stored);
    }
  }, [setTheme]);

  useEffect(() => {
    document.documentElement.setAttribute('data-theme', theme);
    localStorage.setItem('vllm-studio-theme', theme);
  }, [theme]);

  useEffect(() => {
    const mq = window.matchMedia('(prefers-color-scheme: dark)');
    const handler = () => {
      const current = localStorage.getItem('vllm-studio-theme');
      if (!current || current === 'system') {
        document.documentElement.setAttribute('data-theme', 'system');
      }
    };
    mq.addEventListener('change', handler);
    return () => mq.removeEventListener('change', handler);
  }, []);

  const toggle = useCallback(() => {
    const current = useAppStore.getState().theme;
    setTheme(current === 'dark' ? 'light' : 'dark');
  }, [setTheme]);

  const isDark = theme === 'dark' || (theme === 'system' && window.matchMedia('(prefers-color-scheme: dark)').matches);

  return { theme, isDark, toggle, setTheme };
}
