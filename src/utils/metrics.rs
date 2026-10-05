//! Microsecond latency timing and rolling percentiles.

use serde::Serialize;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Mutex;
use std::time::Instant;

#[derive(Debug, Clone, Copy)]
pub struct LatencyTimer {
    start: Instant,
}

impl LatencyTimer {
    pub fn start() -> Self {
        Self {
            start: Instant::now(),
        }
    }

    pub fn elapsed_us(&self) -> u64 {
        self.start.elapsed().as_micros() as u64
    }

    pub fn elapsed_ms(&self) -> f64 {
        self.start.elapsed().as_secs_f64() * 1000.0
    }
}

impl Default for LatencyTimer {
    fn default() -> Self {
        Self::start()
    }
}

/// Number of most recent successful requests kept for percentiles.
const WINDOW: usize = 4096;

    errors: AtomicU64,
}

struct Ring {
    samples: Vec<u64>,
    next: usize,
}

#[derive(Debug, Clone, Serialize)]
pub struct LatencySnapshot {
    pub total_requests: u64,
    pub errors: u64,
    pub window_size: usize,
    pub p50_us: u64,
    pub p95_us: u64,
    pub p99_us: u64,
    pub max_us: u64,
    pub mean_us: f64,
}

impl LatencyStats {
    pub fn new() -> Self {
        Self {
            window: Mutex::new(Ring {
                samples: Vec::with_capacity(WINDOW),
                next: 0,
            }),
            total: AtomicU64::new(0),
            errors: AtomicU64::new(0),
        }
    }

    pub fn record(&self, us: u64) {
        self.total.fetch_add(1, Ordering::Relaxed);
        let mut ring = self.window.lock().unwrap_or_else(|p| p.into_inner());
        if ring.samples.len() < WINDOW {
            ring.samples.push(us);
        } else {
            let i = ring.next;
            ring.samples[i] = us;
            ring.next = (i + 1) % WINDOW;
        }
    }

    pub fn record_error(&self) {
        self.total.fetch_add(1, Ordering::Relaxed);
        self.errors.fetch_add(1, Ordering::Relaxed);
    }

    pub fn snapshot(&self) -> LatencySnapshot {
        let mut sorted = self
            .window
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .samples
            p50_us: percentile(&sorted, 0.50),
            p95_us: percentile(&sorted, 0.95),
            p99_us: percentile(&sorted, 0.99),
            max_us: sorted.last().copied().unwrap_or(0),
            mean_us,
        }
    }
}

impl Default for LatencyStats {
    fn default() -> Self {
        Self::new()
    }
}

pub fn percentile(sorted: &[u64], q: f64) -> u64 {
    if sorted.is_empty() {
        return 0;
    }
    let idx = ((sorted.len() - 1) as f64 * q).round() as usize;
    sorted[idx.min(sorted.len() - 1)]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn percentiles_over_window() {
        let s = LatencyStats::new();
        for us in 1..=100 {
            s.record(us);
        }
        s.record_error();
        let snap = s.snapshot();
        assert_eq!(snap.total_requests, 101);
        assert_eq!(snap.errors, 1);
        assert_eq!(snap.max_us, 100);
        assert!((50..=51).contains(&snap.p50_us));
    }
}
