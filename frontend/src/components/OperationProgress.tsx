import { useEffect, useState } from 'react';
import './OperationProgress.css';

/** Real task text above an animated waiting indicator; no invented percentage. */
export default function OperationProgress({ label }: { label: string }) {
  const [started] = useState(() => Date.now());
  const [seconds, setSeconds] = useState(0);
  useEffect(() => {
    const timer = window.setInterval(() => setSeconds(Math.floor((Date.now() - started) / 1000)), 1000);
    return () => window.clearInterval(timer);
  }, [started]);
  return <div className="operation-progress" aria-busy="true">
    <div className="operation-progress__label">
      <span role="status">{label}</span>
      <span aria-hidden="true">{seconds}s</span>
    </div>
    <div className="operation-progress__track" role="progressbar" aria-label={label}>
      <span />
    </div>
  </div>;
}
