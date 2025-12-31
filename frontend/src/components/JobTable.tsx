import { useEffect, useMemo, useRef, useState } from 'react';
import type { MouseEvent as ReactMouseEvent } from 'react';
import clsx from 'clsx';
import type { Job } from '../types';

interface JobTableProps {
  jobs: Job[];
  isLoading: boolean;
  selectedJobs: Set<string>;
  onToggleJob: (jobId: string, selected: boolean) => void;
  onTogglePage: (jobIds: string[], selected: boolean) => void;
  selectableStatuses: Set<Job['status']>;
}

const PAGE_SIZES = [10, 25, 50, 100];

type SortKey = 'timestamp' | 'stitched_size' | 'expected_size' | 'status';

const SORT_OPTIONS: Array<{ value: SortKey; label: string }> = [
  { value: 'timestamp', label: 'Timestamp' },
  { value: 'stitched_size', label: 'Output Size' },
  { value: 'expected_size', label: 'Expected Size' },
  { value: 'status', label: 'Status' }
];

const statusLabels: Record<Job['status'], string> = {
  unprocessed: 'Queued',
  processing: 'Processing',
  processed: 'Done',
  failed: 'Failed'
};

const statusClasses: Record<Job['status'], string> = {
  unprocessed: 'status-badge queued',
  processing: 'status-badge running',
  processed: 'status-badge success',
  failed: 'status-badge failed'
};

const PREVIEW_SIZE = 512;
const PREVIEW_GAP = 12;

function ThumbnailCell({ url, alt }: { url: string; alt: string }) {
  const wrapperRef = useRef<HTMLDivElement>(null);
  const [previewPos, setPreviewPos] = useState<{ top: number; left: number } | null>(null);

  const handlePointerMove = (event: ReactMouseEvent<HTMLDivElement>) => {
    const wrapper = wrapperRef.current;
    if (!wrapper) {
      return;
    }
    const rect = wrapper.getBoundingClientRect();
    const viewportHeight = window.innerHeight;
    const viewportWidth = window.innerWidth;
    const pointerY = event.clientY;
    const showAbove = viewportHeight - pointerY < PREVIEW_SIZE + PREVIEW_GAP;
    let top = showAbove ? pointerY - PREVIEW_GAP - PREVIEW_SIZE : pointerY + PREVIEW_GAP;
    top = Math.min(Math.max(8, top), viewportHeight - PREVIEW_SIZE - 8);
    const spaceRight = viewportWidth - rect.right;
    const showLeft = spaceRight < PREVIEW_SIZE + PREVIEW_GAP && rect.left - PREVIEW_GAP - PREVIEW_SIZE > 0;
    let left = showLeft ? rect.left - PREVIEW_GAP - PREVIEW_SIZE : rect.right + PREVIEW_GAP;
    left = Math.min(Math.max(8, left), viewportWidth - PREVIEW_SIZE - 8);
    setPreviewPos({ top, left });
  };

  return (
    <div
      className="thumbnail-wrapper"
      ref={wrapperRef}
      onMouseEnter={handlePointerMove}
      onMouseMove={handlePointerMove}
      onMouseLeave={() => setPreviewPos(null)}
    >
      <img className="thumbnail" src={url} alt={alt} />
      {previewPos && (
        <img
          className="thumbnail-preview"
          src={url}
          alt=""
          aria-hidden="true"
          style={{ top: previewPos.top, left: previewPos.left }}
        />
      )}
    </div>
  );
}

const formatBytes = (bytes: number) => {
  if (!bytes) {
    return '0 B';
  }
  const units = ['B', 'KB', 'MB', 'GB', 'TB'];
  let size = bytes;
  let unitIndex = 0;
  while (size >= 1024 && unitIndex < units.length - 1) {
    size /= 1024;
    unitIndex += 1;
  }
  return `${size.toFixed(size >= 10 ? 0 : 1)} ${units[unitIndex]}`;
};

const formatDate = (timestamp: string) => {
  try {
    return new Date(timestamp).toLocaleString();
  } catch {
    return timestamp;
  }
};

export function JobTable({
  jobs,
  isLoading,
  selectedJobs,
  onToggleJob,
  onTogglePage,
  selectableStatuses
}: JobTableProps) {
  const [pageSize, setPageSize] = useState(PAGE_SIZES[0]);
  const [page, setPage] = useState(0);
  const [sortKey, setSortKey] = useState<SortKey>('timestamp');
  const [sortDir, setSortDir] = useState<'asc' | 'desc'>('desc');
  const headerCheckboxRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    setPage(0);
  }, [pageSize, jobs.length, sortKey, sortDir]);

  const sortedJobs = useMemo(() => {
    const multiplier = sortDir === 'asc' ? 1 : -1;
    return [...jobs].sort((a, b) => {
      let result = 0;
      if (sortKey === 'timestamp') {
        const aTime = new Date(a.timestamp).getTime();
        const bTime = new Date(b.timestamp).getTime();
        result = aTime - bTime;
      } else if (sortKey === 'status') {
        result = a.status.localeCompare(b.status);
      } else {
        result = (a[sortKey] ?? 0) - (b[sortKey] ?? 0);
      }
      if (result === 0) {
        return new Date(a.timestamp).getTime() - new Date(b.timestamp).getTime();
      }
      return result * multiplier;
    });
  }, [jobs, sortDir, sortKey]);

  const totalPages = Math.max(1, Math.ceil(sortedJobs.length / pageSize));

  useEffect(() => {
    setPage((prev) => Math.min(prev, totalPages - 1));
  }, [totalPages]);

  const start = page * pageSize;
  const end = Math.min(start + pageSize, sortedJobs.length);
  const paginatedJobs = sortedJobs.slice(start, end);
  const selectablePageIds = paginatedJobs
    .filter((job) => selectableStatuses.has(job.status))
    .map((job) => job.id);
  const allVisibleSelected =
    selectablePageIds.length > 0 && selectablePageIds.every((id) => selectedJobs.has(id));
  const someVisibleSelected =
    selectablePageIds.some((id) => selectedJobs.has(id)) && !allVisibleSelected;

  useEffect(() => {
    if (headerCheckboxRef.current) {
      headerCheckboxRef.current.indeterminate = someVisibleSelected;
    }
  }, [someVisibleSelected, allVisibleSelected, selectablePageIds.length]);

  const visiblePages = useMemo(() => {
    if (totalPages <= 7) {
      return Array.from({ length: totalPages }, (_, i) => i);
    }
    const pages = new Set<number>([0, totalPages - 1, page]);
    if (page > 0) pages.add(page - 1);
    if (page > 1) pages.add(page - 2);
    if (page < totalPages - 1) pages.add(page + 1);
    if (page < totalPages - 2) pages.add(page + 2);
    return Array.from(pages).sort((a, b) => a - b);
  }, [page, totalPages]);

  if (isLoading) {
    return <div className="panel">Loading jobs…</div>;
  }

  if (!jobs.length) {
    return <div className="panel">No jobs recorded yet.</div>;
  }

  return (
    <div className="panel">
      <div className="table-wrapper">
        <div className="table-controls">
          <div className="page-size-select">
            <label htmlFor="page-size">Rows per page:</label>
            <select
              id="page-size"
              value={pageSize}
              onChange={(event) => setPageSize(Number(event.target.value))}
            >
              {PAGE_SIZES.map((size) => (
                <option key={size} value={size}>
                  {size}
                </option>
              ))}
            </select>
          </div>
          <div className="sort-controls">
            <label htmlFor="sort-field">Sort by:</label>
            <select
              id="sort-field"
              value={sortKey}
              onChange={(event) => setSortKey(event.target.value as SortKey)}
            >
              {SORT_OPTIONS.map((option) => (
                <option key={option.value} value={option.value}>
                  {option.label}
                </option>
              ))}
            </select>
            <button
              type="button"
              className="sort-direction"
              onClick={() => setSortDir((prev) => (prev === 'asc' ? 'desc' : 'asc'))}
              aria-label="Toggle sort direction"
            >
              {sortDir === 'asc' ? '↑' : '↓'}
            </button>
          </div>
          <div className="page-info">
            Showing {start + 1}-{end} of {sortedJobs.length}
          </div>
        </div>
        <table>
          <thead>
            <tr>
              <th className="select-column">
                <input
                  ref={headerCheckboxRef}
                  type="checkbox"
                  className="select-checkbox"
                  checked={allVisibleSelected}
                  onChange={(event) => onTogglePage(selectablePageIds, event.target.checked)}
                  disabled={selectablePageIds.length === 0}
                  aria-label="Select visible jobs"
                />
              </th>
              <th>Preview</th>
              <th>Timestamp</th>
              <th>Status</th>
              <th>Progress</th>
              <th>Output</th>
              <th>PID</th>
              <th>Updated</th>
            </tr>
          </thead>
          <tbody>
            {paginatedJobs.map((job) => {
              const percent = Math.max(
                0,
                Math.min(100, Math.round((job.process ?? 0) * 100))
              );
              const selectable = selectableStatuses.has(job.status);
              return (
                <tr key={job.id}>
                  <td className="select-column">
                    <input
                      type="checkbox"
                      className="select-checkbox"
                      checked={selectedJobs.has(job.id)}
                      disabled={!selectable}
                      onChange={(event) => onToggleJob(job.id, event.target.checked)}
                      aria-label={`Select job ${job.timestamp}`}
                    />
                  </td>
                  <td>
                    {job.thumbnail_url ? (
                      <ThumbnailCell url={job.thumbnail_url} alt={`Thumbnail for ${job.timestamp}`} />
                    ) : (
                      <span className="muted">None</span>
                    )}
                  </td>
                  <td className="mono">{job.timestamp}</td>
                  <td>
                    <span className={clsx(statusClasses[job.status])}>
                      {statusLabels[job.status]}
                    </span>
                  </td>
                  <td>
                    <div className="progress">
                      <div className="progress-value" style={{ width: `${percent}%` }} />
                    </div>
                    <small>
                      {formatBytes(job.stitched_size)} / {formatBytes(job.expected_size)} ({percent}% )
                    </small>
                  </td>
                  <td>
                    <div className="mono">{job.final_file}</div>
                  </td>
                  <td className="mono">{job.pid ?? '—'}</td>
                  <td>{formatDate(job.updated_at)}</td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
      <div className="pagination">
        <button
          type="button"
          className="page-arrow"
          onClick={() => setPage((prev) => Math.max(prev - 1, 0))}
          disabled={page === 0}
          aria-label="Previous page"
        >
          ‹
        </button>
        {visiblePages.map((p, index) => {
          const prev = visiblePages[index - 1];
          const needsEllipsis = typeof prev === 'number' && p - prev > 1;
          return (
            <span key={p} className="page-cluster">
              {needsEllipsis && <span className="ellipsis">…</span>}
              <button
                type="button"
                className={clsx('page-button', { active: p === page })}
                onClick={() => setPage(p)}
              >
                {p + 1}
              </button>
            </span>
          );
        })}
        <button
          type="button"
          className="page-arrow"
          onClick={() => setPage((prev) => Math.min(prev + 1, totalPages - 1))}
          disabled={page >= totalPages - 1}
          aria-label="Next page"
        >
          ›
        </button>
      </div>
    </div>
  );
}
