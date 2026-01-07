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

export function generateThumbnailsForJobs(jobIds: string[]): Promise<{ scheduled: string }> {
  return request<{ scheduled: string }>('/tasks', {
    method: 'POST',
    headers,
    body: JSON.stringify({ action: 'generate_thumbnails', job_ids: jobIds })
  });
}

export interface ParallelismPayload {
  stitch_parallelism: number;
  scan_parallelism: number;
  deep_scan_parallelism: number;
  thumbnail_parallelism: number;
}

export function updateParallelism(payload: ParallelismPayload): Promise<ParallelismPayload> {
  return request<ParallelismPayload>('/settings/parallelism', {
    method: 'POST',
    headers,
    body: JSON.stringify(payload)
  });
}

export function updateExpectedRatio(expectedRatio: number): Promise<{ expected_size_ratio: number }> {
  return request<{ expected_size_ratio: number }>('/settings/ratio', {
    method: 'POST',
    headers,
    body: JSON.stringify({ expected_size_ratio: expectedRatio })
  });
}

export function computeExpectedRatio(): Promise<{ expected_size_ratio: number }> {
  return request<{ expected_size_ratio: number }>('/settings/ratio/compute', {
    method: 'POST',
    headers
  });
}

export function updateStitchSettings(settings: {
  output_size: string;
  bitrate: string;
  stitch_type: string;
  auto_resolution: boolean;
  original_bitrate: boolean;
}): Promise<{
  output_size: string;
  bitrate: string;
  stitch_type: string;
  auto_resolution: boolean;
  original_bitrate: boolean;
}> {
  return request<{
    output_size: string;
    bitrate: string;
    stitch_type: string;
    auto_resolution: boolean;
    original_bitrate: boolean;
  }>(
    '/settings/stitch',
    {
      method: 'POST',
      headers,
      body: JSON.stringify(settings)
    }
  );
}
