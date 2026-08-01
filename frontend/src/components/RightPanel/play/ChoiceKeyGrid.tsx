import { pickedLabelStyle } from './paneKeys';

interface Choice {
  id: string | number;
  label: string;
  ariaLabel?: string;
  title: string;
  checked: boolean;
  onSelect: () => void;
}

interface ChoiceKeyGridProps {
  choices: Choice[];
  /** Full width (w-full), as the Drums grid has. */
  fullWidth?: boolean;
}

const COLUMNS = 4;

/** Radio keys four to a row; the rows share the pane's remaining height,
 *  each at least as tall as an sm key (26px). Labels wrap inside the key. */
export default function ChoiceKeyGrid({ choices, fullWidth }: ChoiceKeyGridProps) {
  return (
    <div className={fullWidth ? 'grid min-h-0 w-full flex-1 gap-[3px]' : 'grid min-h-0 flex-1 gap-[3px]'}
      style={{
        gridTemplateColumns: `repeat(${COLUMNS}, minmax(0, 1fr))`,
        gridTemplateRows: `repeat(${Math.ceil(choices.length / COLUMNS)}, minmax(26px, 1fr))`,
      }}
    >
      {choices.map((choice) => (
        <button key={choice.id} type="button" role="radio" aria-label={choice.ariaLabel}
          aria-checked={choice.checked} tabIndex={choice.checked ? 0 : -1}
          data-active={choice.checked ? '' : undefined} onClick={choice.onSelect} title={choice.title}
          className="te-key btn-hover h-full min-w-0"
          style={{ minHeight: 26, padding: '0 3px', fontSize: 12, lineHeight: '12px', letterSpacing: 'normal',
            whiteSpace: 'normal', overflowWrap: 'anywhere', ...pickedLabelStyle(choice.checked) }}>
          {choice.label}
        </button>
      ))}
    </div>
  );
}
