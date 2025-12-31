import { useEffect, useMemo, useState } from 'react';
import clsx from 'clsx';
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import {
  fetchStatus,
  generateThumbnailsForJobs,
  stitchSelectedJobs,
  triggerTask,
  updateParallelism
} from './api';
import type { Job, TaskAction } from './types';
import { JobTable } from './components/JobTable';
import './App.css';

const ACTIONS: Array<{
  action: TaskAction;
  label: string;
  description: string;
}> = [
  {
    action: 'scan',
    label: 'Scan',
    description: 'Quick scan for new RAW pairs'
  },
  {
    action: 'deep_scan',
    label: 'Deep Scan',
    description: 'Re-evaluate every job and recalculate progress'
  },
  {
    action: 'stitch',
    label: 'Stitch Pending',
    description: 'Run stitcher for queued items only'
  },
  {
    action: 'full_stitch',
    label: 'Retry Failed',
    description: 'Attempt stitching for failed jobs as well'
  },
  {
    action: 'generate_thumbnails',
    label: 'Generate Thumbnails',
    description: 'Extract preview thumbnails for jobs missing them'
  }
];

const SELECTABLE_STATUSES = new Set<Job['status']>(['unprocessed', 'failed']);

const statusOrder: Record<TaskAction | 'status', number> = {
  scan: 0,
  deep_scan: 1,
  stitch: 2,
  full_stitch: 3,
  generate_thumbnails: 4,
  stitch_selected: 5,
  status: 4
};

export default function App() {
  const queryClient = useQueryClient();
  const statusQuery = useQuery({
    queryKey: ['status'],
    queryFn: fetchStatus,
    refetchInterval: 15000
  });

  const mutation = useMutation({
    mutationFn: (action: TaskAction) => triggerTask(action),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['status'] });
    }
  });

  const jobs = statusQuery.data?.jobs ?? [];
  const [selectedJobs, setSelectedJobs] = useState<Set<string>>(new Set());
  const [settingsOpen, setSettingsOpen] = useState(false);
  const [parallelValue, setParallelValue] = useState(1);

  useEffect(() => {
    setSelectedJobs((prev) => {
      const validIds = new Set(
        jobs.filter((job) => SELECTABLE_STATUSES.has(job.status)).map((job) => job.id)
      );
      const next = new Set<string>();
      prev.forEach((id) => {
        if (validIds.has(id)) {
          next.add(id);
        }
      });
      return next.size === prev.size ? prev : next;
    });
  }, [jobs]);

  const parallelMutation = useMutation({
    mutationFn: (value: number) => updateParallelism(value),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['status'] });
      setSettingsOpen(false);
    }
  });

  const stitchSelectedMutation = useMutation({
    mutationFn: (jobIds: string[]) => stitchSelectedJobs(jobIds),
    onSuccess: () => {
      setSelectedJobs(new Set());
      queryClient.invalidateQueries({ queryKey: ['status'] });
    }
  });
  const thumbnailsSelectedMutation = useMutation({
    mutationFn: (jobIds: string[]) => generateThumbnailsForJobs(jobIds),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['status'] });
    }
  });
  const summary = useMemo(() => summarizeJobs(jobs), [jobs]);
  const activeJobs = statusQuery.data?.active_jobs ?? [];
  const pendingJobs = statusQuery.data?.pending_jobs ?? 0;
  const maxParallelJobs = statusQuery.data?.max_parallel_jobs ?? 1;

  useEffect(() => {
    if (settingsOpen) {
      setParallelValue(maxParallelJobs);
    }
  }, [settingsOpen, maxParallelJobs]);
  const lastUpdated = statusQuery.dataUpdatedAt ? new Date(statusQuery.dataUpdatedAt).toLocaleTimeString() : '—';

  const trigger = (action: TaskAction) => {
    mutation.mutate(action);
  };

  return (
    <div className="app-container">
      <header className="panel hero">
        <div>
          <h1>Insta360 Autostitcher</h1>
          <p className="muted">Monitor queued jobs and trigger scans or stitching from the browser.</p>
        </div>
        <div className="hero-stats">
          <div>
            <span className="stat-label">Total Jobs</span>
            <span className="stat-value">{summary.total}</span>
          </div>
          <div>
            <span className="stat-label">Active Threads</span>
            <span className="stat-value">{activeJobs.length}</span>
          </div>
          <div>
            <span className="stat-label">Parallel Jobs</span>
            <span className="stat-value">{maxParallelJobs}</span>
          </div>
          <div>
            <span className="stat-label">Last Updated</span>
            <span className="stat-value mono">{lastUpdated}</span>
          </div>
        </div>
      </header>

      <section className="panel actions">
        <div className="actions-header">
          <h2>Controls</h2>
          <div className="action-buttons">
            <button className="ghost" type="button" onClick={() => statusQuery.refetch()} disabled={statusQuery.isFetching}>
              Refresh
            </button>
            <button className="ghost" type="button" onClick={() => setSettingsOpen(true)}>
              Settings
            </button>
          </div>
        </div>
        <div className="action-grid">
          {ACTIONS.map(({ action, label, description }) => (
            <button
              key={action}
              type="button"
              className="action-card"
              disabled={mutation.isPending}
              onClick={() => trigger(action)}
            >
              <div className="action-card-content">
                <span className="action-label">{label}</span>
                <span className="action-description">{description}</span>
              </div>
            </button>
          ))}
        </div>
        {mutation.isPending && <p className="muted">Sending command…</p>}
        {mutation.isError && <p className="error">{(mutation.error as Error)?.message}</p>}
      </section>

      <section className="panel summary">
        <h2>Job Summary</h2>
        <div className="summary-grid">
          <SummaryCard label="Queued" value={summary.unprocessed} tone="queued" />
          <SummaryCard label="Processing" value={summary.processing} tone="running" />
          <SummaryCard label="Finished" value={summary.processed} tone="success" />
          <SummaryCard label="Failed" value={summary.failed} tone="failed" />
        </div>
      </section>

      {statusQuery.isError && (
        <div className="panel error">Failed to load status: {(statusQuery.error as Error)?.message}</div>
      )}

      <JobTable
        jobs={jobs}
        isLoading={statusQuery.isLoading}
        selectedJobs={selectedJobs}
        selectableStatuses={SELECTABLE_STATUSES}
        onToggleJob={(jobId, selected) =>
          setSelectedJobs((prev) => {
            const next = new Set(prev);
            if (selected) {
              next.add(jobId);
            } else {
              next.delete(jobId);
            }
            return next;
          })
        }
        onTogglePage={(jobIds, selected) =>
          setSelectedJobs((prev) => {
            const next = new Set(prev);
            jobIds.forEach((id) => {
              if (selected) {
                next.add(id);
              } else {
                next.delete(id);
              }
            });
            return next;
          })
        }
      />
      <div className="selection-actions">
        <span>
          {selectedJobs.size} selected · {pendingJobs} waiting
        </span>
        <div className="selection-buttons">
          <button
            type="button"
            className="ghost"
            onClick={() => thumbnailsSelectedMutation.mutate(Array.from(selectedJobs))}
            disabled={selectedJobs.size === 0 || thumbnailsSelectedMutation.isPending}
          >
            {thumbnailsSelectedMutation.isPending ? 'Generating…' : 'Thumbnails Selected'}
          </button>
          <button
            type="button"
            className="primary"
            onClick={() => stitchSelectedMutation.mutate(Array.from(selectedJobs))}
            disabled={selectedJobs.size === 0 || stitchSelectedMutation.isPending}
          >
            {stitchSelectedMutation.isPending ? 'Stitching…' : 'Stitch Selected'}
          </button>
        </div>
      </div>
      {thumbnailsSelectedMutation.isError && (
        <div className="panel error">
          Failed to generate thumbnails: {(thumbnailsSelectedMutation.error as Error)?.message}
        </div>
      )}
      {stitchSelectedMutation.isError && (
        <div className="panel error">Failed to stitch selected: {(stitchSelectedMutation.error as Error)?.message}</div>
      )}
      {settingsOpen && (
        <div className="modal-backdrop">
          <div className="modal">
            <h3>Settings</h3>
            <label htmlFor="parallel-input">Parallel jobs</label>
            <input
              id="parallel-input"
              type="number"
              min={1}
              value={parallelValue}
              onChange={(event) => {
                const next = Number(event.target.value);
                setParallelValue(Number.isNaN(next) ? 1 : Math.max(1, Math.round(next)));
              }}
            />
            <div className="modal-actions">
              <button type="button" className="ghost" onClick={() => setSettingsOpen(false)} disabled={parallelMutation.isPending}>
                Cancel
              </button>
              <button
                type="button"
                className="primary"
                onClick={() => parallelMutation.mutate(parallelValue)}
                disabled={parallelMutation.isPending}
              >
                {parallelMutation.isPending ? 'Saving…' : 'Save'}
              </button>
            </div>
            {parallelMutation.isError && (
              <p className="error">Failed to update parallelism: {(parallelMutation.error as Error)?.message}</p>
            )}
          </div>
        </div>
      )}
    </div>
  );
}

function summarizeJobs(jobs: Job[]) {
  return jobs.reduce(
    (acc, job) => {
      acc.total += 1;
      acc[job.status] += 1;
      return acc;
    },
    { total: 0, unprocessed: 0, processing: 0, processed: 0, failed: 0 }
  );
}

interface SummaryCardProps {
  label: string;
  value: number;
  tone: 'queued' | 'running' | 'success' | 'failed';
}

function SummaryCard({ label, value, tone }: SummaryCardProps) {
  return (
    <div className={clsx('summary-card', tone)}>
      <span className="muted">{label}</span>
      <span className="stat-value">{value}</span>
    </div>
  );
}
