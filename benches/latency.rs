//! End-to-end decision latency (tokenize + prefill + softmax).
//!
//!   KRONOS_MODEL_DIR=./models/qwen2.5-0.5b-instruct cargo bench --bench latency
//!   (add --features cuda or --features metal for GPU)

use kronos::utils::metrics::percentile;
use kronos::{DecisionRequest, EmbeddedKronos};
use std::time::Instant;

fn main() -> anyhow::Result<()> {
    let model_dir = std::env::var("KRONOS_MODEL_DIR")
        .unwrap_or_else(|_| "./models/qwen2.5-0.5b-instruct".to_owned());
    let iterations: usize = std::env::var("KRONOS_BENCH_ITERS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(500);

    let kronos = EmbeddedKronos::new(&model_dir)?;
    println!(
        "model={:?} device={} format={:?}",
        kronos.model_kind(),
        kronos.device_label(),
        kronos.prompt_format()
    );

    let req = DecisionRequest {
        prompt: "Market volatility: high. Order size: 50,000 USD. Account risk limit: 40,000 USD.".to_owned(),
        choices: vec!["buy".into(), "sell".into(), "hold".into()],
        temperature: None,
        min_confidence: None,
    };

    // Warmup: kernel compilation, allocator, choice cache
    for _ in 0..10 {
        kronos.evaluate(&req)?;
    }

    let mut samples = Vec::with_capacity(iterations);
    let mut last = None;
    for _ in 0..iterations {
        let t = Instant::now();
        last = Some(kronos.evaluate(&req)?);
        samples.push(t.elapsed().as_micros() as u64);
    }
    samples.sort_unstable();

    let ms = |us: u64| us as f64 / 1000.0;
    println!("=== Kronos latency, {iterations} iterations ===");
    println!("p50: {:>8.3} ms", ms(percentile(&samples, 0.50)));
    println!("p95: {:>8.3} ms", ms(percentile(&samples, 0.95)));
    println!("p99: {:>8.3} ms", ms(percentile(&samples, 0.99)));
    println!("max: {:>8.3} ms", ms(*samples.last().unwrap_or(&0)));
    if let Some(d) = last {
        println!(
            "last decision: {} (confidence {:.3}, choice_mass {:.3}, {} prompt tokens)",
            d.selected_choice, d.confidence, d.choice_mass, d.prompt_tokens
        );
    }
    Ok(())
}
