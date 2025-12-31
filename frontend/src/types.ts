export type JobStatus = 'unprocessed' | 'processing' | 'processed' | 'failed';

export interface Job {
  id: string;
  timestamp: string;
  final_file: string;
  source_files: string[];
  status: JobStatus;
  pid: number | null;
  stitched_size: number;
  process: number;
  expected_size: number;
  created_at: string;
  updated_at: string;
}

export interface StatusResponse {
  jobs: Job[];
  active_jobs: string[];
}

export type TaskAction = 'scan' | 'deep_scan' | 'stitch' | 'full_stitch' | 'generate_thumbnails';
