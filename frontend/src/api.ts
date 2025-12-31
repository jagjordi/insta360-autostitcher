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
