import { useEffect, useRef } from 'react';
import { createPortal } from 'react-dom';

export interface ContextMenuItem {
  id: string;
  label: string;
  /** Keyboard shortcut shown beside the label. */
  hint?: string;
  disabled?: boolean;
  /** Why the item is disabled, as its tooltip. */
  title?: string;
  /** Draw a separator above this item. */
  divider?: boolean;
  run: () => void;
}

interface Props {
  x: number;
  y: number;
  items: ContextMenuItem[];
  onChoose: (item: ContextMenuItem) => void;
  onClose: () => void;
}

const MENU_WIDTH = 208;

/** Right-click menu over the selected passage: clipboard edits on the
 *  selection, then the tools. Opening a tool never starts a model call. */
export default function ScoreContextMenu({ x, y, items, onChoose, onClose }: Props) {
  const menu = useRef<HTMLDivElement>(null);
  useEffect(() => {
    const element = menu.current;
    const buttons = [...element?.querySelectorAll<HTMLButtonElement>('button:not([disabled])') ?? []];
    buttons[0]?.focus();
    const dismissOutside = (event: PointerEvent) => {
      if (!element?.contains(event.target as Node)) onClose();
    };
    const keyboard = (event: KeyboardEvent) => {
      if (event.key === 'Escape' || event.key === 'Tab') {
        if (event.key === 'Escape') event.preventDefault();
        onClose();
      } else if (['ArrowDown', 'ArrowUp', 'Home', 'End'].includes(event.key)) {
        event.preventDefault();
        const current = buttons.indexOf(document.activeElement as HTMLButtonElement);
        const next = event.key === 'Home' ? 0 : event.key === 'End' ? buttons.length - 1
          : (current + (event.key === 'ArrowDown' ? 1 : -1) + buttons.length) % buttons.length;
        buttons[next]?.focus();
      }
    };
    window.addEventListener('pointerdown', dismissOutside);
    window.addEventListener('keydown', keyboard);
    window.addEventListener('scroll', onClose, true);
    window.addEventListener('resize', onClose);
    return () => {
      window.removeEventListener('pointerdown', dismissOutside);
      window.removeEventListener('keydown', keyboard);
      window.removeEventListener('scroll', onClose, true);
      window.removeEventListener('resize', onClose);
    };
  }, [onClose]);

  const height = items.length * 34 + 12;
  return createPortal(<div ref={menu} role="menu" aria-label="Selected passage" className="fixed p-1 rounded-md shadow-xl"
    onContextMenu={event => event.preventDefault()}
    style={{ left: Math.max(8, Math.min(x, window.innerWidth - MENU_WIDTH - 8)),
      top: Math.max(8, Math.min(y, window.innerHeight - height - 8)), width: MENU_WIDTH, maxWidth: 'calc(100vw - 16px)',
      zIndex: 100, background: 'var(--bg-surface)', color: 'var(--text-primary)', border: '1px solid var(--border)' }}>
    {items.map(item => <button key={item.id} role="menuitem" disabled={item.disabled} title={item.disabled ? item.title : undefined}
      onClick={() => onChoose(item)}
      className="flex w-full items-center justify-between gap-3 px-3 py-2 text-left text-sm rounded btn-hover focus-visible:outline-2 disabled:opacity-40 disabled:cursor-default"
      style={{ outlineColor: 'var(--brand-yellow)', ...(item.divider ? { borderTop: '1px solid var(--border)', borderRadius: 0, marginTop: 4, paddingTop: 10 } : {}) }}>
      <span>{item.label}</span>
      {item.hint && <span className="text-xs" style={{ color: 'var(--text-muted)' }}>{item.hint}</span>}
    </button>)}
  </div>, document.body);
}
