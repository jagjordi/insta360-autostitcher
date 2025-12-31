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
  thumbnail_url: string | null;
}

export interface StatusResponse {
  jobs: Job[];
  active_jobs: string[];
  pending_jobs: number;
  max_parallel_jobs: number;
  expected_size_ratio: number;
  stitch_settings: {
    output_size: string;
    bitrate: string;
    stitch_type: string;
    auto_resolution: boolean;
  };
  concurrency?: {
    stitch: number;
    scan: number;
    deep_scan: number;
  };
}

export type TaskAction = 'scan' | 'deep_scan' | 'stitch' | 'full_stitch' | 'generate_thumbnails' | 'stitch_selected';
