import type { StatusResponse, TaskAction } from './types';

const API_BASE = (import.meta.env.VITE_API_BASE as string | undefined) ?? '';

const headers = {
  'Content-Type': 'application/json'
};

async function request<T>(path: string, options?: RequestInit): Promise<T> {
  const response = await fetch(`${API_BASE}${path}`, options);
  if (!response.ok) {
    const message = await response.text();
    throw new Error(message || 'Request failed');
  }
  return response.json() as Promise<T>;
}

export function fetchStatus(): Promise<StatusResponse> {
  return request<StatusResponse>('/status', {
    headers
  });
}

export function triggerTask(action: TaskAction): Promise<{ scheduled: string }> {
  return request<{ scheduled: string }>('/tasks', {
    method: 'POST',
    headers,
    body: JSON.stringify({ action })
  });
}

export function stitchSelectedJobs(jobIds: string[]): Promise<{ scheduled: string }> {
  return request<{ scheduled: string }>('/tasks', {
    method: 'POST',
    headers,
    body: JSON.stringify({ action: 'stitch_selected', job_ids: jobIds })
  });
}

export function updateParallelism(maxParallelJobs: number): Promise<{ max_parallel_jobs: number }> {
  return request<{ max_parallel_jobs: number }>('/settings/parallelism', {
    method: 'POST',
    headers,
    body: JSON.stringify({ max_parallel_jobs: maxParallelJobs })
  });
}
