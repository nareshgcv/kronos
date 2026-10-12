//! Latency helpers.

use std::time::Instant;

#[derive(Debug, Clone, Copy)]
pub struct LatencyTimer(Instant);

impl LatencyTimer {
    pub fn start() -> Self {
        Self(Instant::now())
    }

    pub fn elapsed_us(&self) -> u64 {
        self.0.elapsed().as_micros() as u64
    }
}

/// Nearest-rank percentile of an ascending-sorted slice; 0 if empty.
pub fn percentile(sorted: &[u64], q: f64) -> u64 {
    if sorted.is_empty() {
        return 0;
    }
    let rank = (q.clamp(0.0, 1.0) * sorted.len() as f64).ceil() as usize;
    sorted[rank.saturating_sub(1).min(sorted.len() - 1)]
}

#[cfg(test)]
mod tests {
    use super::percentile;

    #[test]
    fn nearest_rank() {
        let s = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
        assert_eq!(percentile(&s, 0.5), 5);
        assert_eq!(percentile(&s, 0.95), 10);
        assert_eq!(percentile(&s, 0.0), 1);
        assert_eq!(percentile(&[], 0.5), 0);
    }
}
