import type { CSSProperties, ReactNode } from 'react';

interface ScreenProps {
  /** The screen's lines. */
  children: ReactNode;
  style?: CSSProperties;
}

/**
 * LCD-style screen, dark glass in both themes: 11px device type on 14px
 * lines, no wrapping. Give it a fixed height from the pane's budget — what
 * does not fit is clipped, never scrolled.
 */
export default function Screen({ children, style }: ScreenProps) {
  return (
    <div
      className="te-screen min-w-0 overflow-hidden whitespace-nowrap"
      style={{ flexShrink: 0, padding: '6px 8px', fontSize: 11, lineHeight: '14px', ...style }}
    >
      {children}
    </div>
  );
}
