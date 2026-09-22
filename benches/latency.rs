use std::time::Instant;
use hdrhistogram::Histogram;
use kronos_core::{KronosEngine, Schema};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let engine = KronosEngine::init()?; // Metal or CUDA context
    let schema = Schema::choices(&["APPROVE", "REJECT", "HALT"]);
    let prompt = "Market volatility: high. Order size: 50,000 USD.";

    // Warmup pass (forces GPU shader compilation & pipeline sync)
    let _ = engine.evaluate(prompt, &schema)?;

    let iterations = 10_000;
    let mut histogram = Histogram::<u64>::new_with_bounds(1, 1_000_000, 3)?; // Microseconds

    for _ in 0..iterations {
        let start = Instant::now();
        let _decision = engine.evaluate(prompt, &schema)?;
        let duration = start.elapsed().as_micros() as u64;
        histogram.record(duration)?;
    }

    println!("=== Kronos Core Latency Profile (Microseconds) ===");
    println!("p50:   {} µs ({:.2} ms)", histogram.value_at_quantile(0.50), histogram.value_at_quantile(0.50) as f64 / 1000.0);
    println!("p95:   {} µs ({:.2} ms)", histogram.value_at_quantile(0.95), histogram.value_at_quantile(0.95) as f64 / 1000.0);
    println!("p99:   {} µs ({:.2} ms)", histogram.value_at_quantile(0.99), histogram.value_at_quantile(0.99) as f64 / 1000.0);
    println!("Max:   {} µs ({:.2} ms)", histogram.max(), histogram.max() as f64 / 1000.0);

    Ok(())
}
