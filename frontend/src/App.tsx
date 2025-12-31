import { useMemo } from 'react';
import clsx from 'clsx';
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import { fetchStatus, triggerTask } from './api';
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

const statusOrder: Record<TaskAction | 'status', number> = {
  scan: 0,
  deep_scan: 1,
  stitch: 2,
  full_stitch: 3,
  generate_thumbnails: 4,
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
  const sortedJobs = useMemo(() => [...jobs].sort((a, b) => (a.timestamp < b.timestamp ? 1 : -1)), [jobs]);
  const summary = useMemo(() => summarizeJobs(jobs), [jobs]);
  const activeJobs = statusQuery.data?.active_jobs ?? [];
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
            <span className="stat-label">Last Updated</span>
            <span className="stat-value mono">{lastUpdated}</span>
          </div>
        </div>
      </header>

      <section className="panel actions">
        <div className="actions-header">
          <h2>Controls</h2>
          <button className="ghost" type="button" onClick={() => statusQuery.refetch()} disabled={statusQuery.isFetching}>
            Refresh
          </button>
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

      <JobTable jobs={sortedJobs} isLoading={statusQuery.isLoading} />
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
