import { create } from 'zustand';

type CleanupTarget = 'current score' | 'saved projects';

interface ImportCleanupState {
  failedTargets: CleanupTarget[];
  failed: (target: CleanupTarget) => void;
  complete: (target: CleanupTarget) => void;
}

/** Only failed source-detail cleanup is reported; recovered musical work stays usable. */
export const useImportCleanupStore = create<ImportCleanupState>((set) => ({
  failedTargets: [],
  failed: (target) => set((state) => state.failedTargets.includes(target) ? state
    : { failedTargets: [...state.failedTargets, target] }),
  complete: (target) => set((state) => state.failedTargets.includes(target)
    ? { failedTargets: state.failedTargets.filter((item) => item !== target) } : state),
}));
