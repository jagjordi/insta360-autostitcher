import { useEffect, useMemo, useState } from 'react';
import clsx from 'clsx';
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import {
  fetchStatus,
  generateThumbnailsForJobs,
  stitchSelectedJobs,
  triggerTask,
  updateExpectedRatio,
  updateParallelism,
  computeExpectedRatio,
  updateStitchSettings
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
  const [ratioValue, setRatioValue] = useState(1);
  const [outputSizeValue, setOutputSizeValue] = useState('');
  const [bitrateValue, setBitrateValue] = useState('');
  const [stitchTypeValue, setStitchTypeValue] = useState('');
  const [autoResolution, setAutoResolution] = useState(false);

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

  const settingsMutation = useMutation({
    mutationFn: async () => {
      const sanitizedOutput = outputSizeValue.trim();
      const sanitizedBitrate = bitrateValue.trim();
      const sanitizedStitch = stitchTypeValue.trim();
      await Promise.all([
        updateParallelism(parallelValue),
        updateExpectedRatio(ratioValue),
        updateStitchSettings({
          output_size: sanitizedOutput,
          bitrate: sanitizedBitrate,
          stitch_type: sanitizedStitch,
          auto_resolution: autoResolution
        })
      ]);
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['status'] });
      setSettingsOpen(false);
    }
  });

  const computeRatioMutation = useMutation({
    mutationFn: () => computeExpectedRatio(),
    onSuccess: (data) => {
      setRatioValue(data.expected_size_ratio);
      queryClient.invalidateQueries({ queryKey: ['status'] });
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
  const expectedRatio = statusQuery.data?.expected_size_ratio ?? 1;
  const stitchSettings = statusQuery.data?.stitch_settings;

  useEffect(() => {
    if (settingsOpen) {
      setParallelValue(maxParallelJobs);
      setRatioValue(expectedRatio);
      setOutputSizeValue(stitchSettings?.output_size ?? '');
      setBitrateValue(stitchSettings?.bitrate ?? '');
      setStitchTypeValue(stitchSettings?.stitch_type ?? '');
      setAutoResolution(stitchSettings?.auto_resolution ?? false);
    }
  }, [settingsOpen, maxParallelJobs, expectedRatio, stitchSettings]);
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
              disabled={settingsMutation.isPending}
            />
            <label htmlFor="ratio-input">Expected size ratio</label>
            <div className="ratio-controls">
              <input
                id="ratio-input"
                type="number"
                min={0.01}
                step={0.01}
                value={ratioValue}
                onChange={(event) => {
                  const next = Number(event.target.value);
                  setRatioValue(Number.isNaN(next) ? 1 : Math.max(0.01, next));
                }}
                disabled={settingsMutation.isPending || computeRatioMutation.isPending}
              />
              <button
                type="button"
                className="ghost"
                onClick={() => computeRatioMutation.mutate()}
                disabled={computeRatioMutation.isPending || settingsMutation.isPending}
              >
                {computeRatioMutation.isPending ? 'Computing…' : 'Compute Ratio'}
              </button>
            </div>
            <div className="checkbox-row">
              <label htmlFor="auto-resolution">
                <input
                  id="auto-resolution"
                  type="checkbox"
                  checked={autoResolution}
                  onChange={(event) => setAutoResolution(event.target.checked)}
                  disabled={settingsMutation.isPending}
                />
                Use default resolution (derive from input)
              </label>
            </div>
            <label htmlFor="resolution-input">Output resolution</label>
            <input
              id="resolution-input"
              type="text"
              value={autoResolution ? '' : outputSizeValue}
              onChange={(event) => setOutputSizeValue(event.target.value)}
              disabled={settingsMutation.isPending || autoResolution}
              placeholder={autoResolution ? 'Auto: 2×input width × input height' : 'e.g. 5760x2880'}
            />
            <label htmlFor="bitrate-input">Bitrate</label>
            <input
              id="bitrate-input"
              type="text"
              value={bitrateValue}
              onChange={(event) => setBitrateValue(event.target.value)}
              disabled={settingsMutation.isPending}
            />
            <label htmlFor="stitch-type-input">Stitch type</label>
            <input
              id="stitch-type-input"
              type="text"
              value={stitchTypeValue}
              onChange={(event) => setStitchTypeValue(event.target.value)}
              disabled={settingsMutation.isPending}
            />
            <div className="modal-actions">
              <button
                type="button"
                className="ghost"
                onClick={() => setSettingsOpen(false)}
                disabled={settingsMutation.isPending}
              >
                Cancel
              </button>
              <button
                type="button"
                className="primary"
                onClick={() => settingsMutation.mutate()}
                disabled={settingsMutation.isPending}
              >
                {settingsMutation.isPending ? 'Saving…' : 'Save'}
              </button>
            </div>
            {computeRatioMutation.isError && (
              <p className="error">
                Failed to compute ratio: {(computeRatioMutation.error as Error)?.message}
              </p>
            )}
            {settingsMutation.isError && (
              <p className="error">Failed to update settings: {(settingsMutation.error as Error)?.message}</p>
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
