import clsx from 'clsx';
import type { Job } from '../types';

interface JobTableProps {
  jobs: Job[];
  isLoading: boolean;
}

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

export function JobTable({ jobs, isLoading }: JobTableProps) {
  if (isLoading) {
    return <div className="panel">Loading jobs…</div>;
  }

  if (!jobs.length) {
    return <div className="panel">No jobs recorded yet.</div>;
  }

  return (
    <div className="panel">
      <div className="table-wrapper">
        <table>
          <thead>
            <tr>
              <th>Timestamp</th>
              <th>Status</th>
              <th>Progress</th>
              <th>Output</th>
              <th>PID</th>
              <th>Updated</th>
            </tr>
          </thead>
          <tbody>
            {jobs.map((job) => {
              const percent = Math.max(
                0,
                Math.min(100, Math.round((job.process ?? 0) * 100))
              );
              return (
                <tr key={job.id}>
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
    </div>
  );
}
