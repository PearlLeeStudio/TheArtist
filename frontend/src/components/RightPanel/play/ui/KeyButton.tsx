import type { CSSProperties, ReactNode } from 'react';

type KeySize = 'sm' | 'md';

interface KeyButtonProps {
  label?: string;
  /** An icon in place of the label (the label then becomes the accessible name). */
  children?: ReactNode;
  /** Latched / selected: pressed look, plus aria-pressed (aria-checked or
   *  aria-selected when `role` is a checkable or selectable role). */
  active?: boolean;
  onClick?: () => void;
  title?: string;
  ariaLabel?: string;
  /** Fixed heights: sm 26px, md 30px (default). */
  size?: KeySize;
  role?: string;
  /** For a tab in a roving-tabindex tablist: -1 takes it out of the tab order. */
  tabIndex?: number;
  id?: string;
  ariaControls?: string;
  className?: string;
  style?: CSSProperties;
}

/** Geometry per size (px): key height, label type and line. */
const SIZES: Record<KeySize, { height: number; font: number; line: number }> = {
  sm: { height: 26, font: 9, line: 10 },
  md: { height: 30, font: 10, line: 12 },
};
/** Side padding inside the 1px border. */
const PAD_X = 3;

const CHECKED_ROLES = new Set(['checkbox', 'radio', 'switch', 'menuitemcheckbox', 'menuitemradio']);
const SELECTED_ROLES = new Set(['tab', 'option', 'gridcell', 'row', 'treeitem']);

/**
 * Square device key: uppercase device type on one line (ellipsis), or an
 * icon. Fills its grid cell's width; its height is fixed by `size`.
 */
export default function KeyButton({
  label,
  children,
  active,
  onClick,
  title,
  ariaLabel,
  size = 'md',
  role,
  tabIndex,
  id,
  ariaControls,
  className,
  style,
}: KeyButtonProps) {
  const geometry = SIZES[size];
  const toggle = role === undefined || role === 'button';
  const checkable = role !== undefined && CHECKED_ROLES.has(role);
  const selectable = role !== undefined && SELECTED_ROLES.has(role);

  return (
    <button
      type="button"
      id={id}
      role={role}
      tabIndex={tabIndex}
      aria-controls={ariaControls}
      aria-pressed={toggle ? active : undefined}
      aria-checked={checkable ? Boolean(active) : undefined}
      aria-selected={selectable ? Boolean(active) : undefined}
      aria-label={ariaLabel ?? (children !== undefined ? label : undefined)}
      title={title}
      onClick={onClick}
      data-active={active ? '' : undefined}
      className={`te-key btn-hover relative flex w-full min-w-0 flex-col items-center justify-center uppercase select-none${className ? ` ${className}` : ''}`}
      style={{ height: geometry.height, paddingInline: PAD_X, fontSize: geometry.font, ...style }}
    >
      {children ?? (label !== undefined && (
        <span className="relative block max-w-full truncate" style={{ lineHeight: `${geometry.line}px` }}>
          {label}
        </span>
      ))}
    </button>
  );
}
