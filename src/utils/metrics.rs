use std::time::Instant;

#[derive(Debug, Clone, Copy)]
pub struct LatencyTimer {
    start: Instant,
}

impl LatencyTimer {
    /// Starts a new latency timer instance.
    pub fn start() -> Self {
        Self {
            start: Instant::now(),
        }
    }

    /// Returns the elapsed time in milliseconds as an `f64`.
    pub fn elapsed_ms(&self) -> f64 {
        self.start.elapsed().as_secs_f64() * 1000.0
    }

    /// Returns the elapsed time in microseconds as an `f64`.
    pub fn elapsed_us(&self) -> f64 {
        self.start.elapsed().as_secs_f64() * 1_000_000.0
    }
}

impl Default for LatencyTimer {
    fn default() -> Self {
        Self::start()
    }
}
