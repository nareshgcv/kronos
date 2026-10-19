//! End-to-end decision latency (tokenize + prefill + softmax), cycling through
//! every prompt in the evaluation datasets.
//!
//!   KRONOS_MODEL_DIR=./models/Qwen3-0.6B cargo bench --bench latency
//!
//! Optional settings:
//!   KRONOS_BENCH_ITERS=2000              number of timed passes (default 10000)
//!   KRONOS_BENCH_DATASETS=eval/datasets  folder of dataset JSON files
//!   (add --features cuda or --features metal for GPU)

use anyhow::{bail, Context, Result};
use kronos::utils::metrics::percentile;
use kronos::{DecisionRequest, EmbeddedKronos, KronosConfig};
use serde_json::Value;
use std::collections::{BTreeMap, HashSet};
use std::path::PathBuf;
use std::time::Instant;

const WARMUP_PASSES: usize = 20;

struct Prompt {
    dataset: String,
    req: DecisionRequest,
}

fn env_or(key: &str, default: &str) -> String {
    std::env::var(key).unwrap_or_else(|_| default.to_owned())
}

/// Builds one request per dataset case, using the same "labels with
/// definitions" prompt layout as the evaluation harness.
fn load_prompts(dir: &str) -> Result<Vec<Prompt>> {
    let mut files: Vec<PathBuf> = std::fs::read_dir(dir)
        .with_context(|| format!("cannot read dataset folder {dir}"))?
        .filter_map(|entry| entry.ok().map(|e| e.path()))
        .filter(|path| path.extension().is_some_and(|ext| ext == "json"))
        .collect();
    files.sort();

    let mut prompts = Vec::new();
    for path in files {
        let text = std::fs::read_to_string(&path)
            .with_context(|| format!("cannot read {}", path.display()))?;
        let ds: Value = serde_json::from_str(&text)
            .with_context(|| format!("invalid JSON in {}", path.display()))?;

        let name = ds["task"].as_str().unwrap_or("unknown").to_owned();
        let labels = ds["labels"].as_array().cloned().unwrap_or_default();
        let choices: Vec<String> = labels
            .iter()
            .filter_map(|l| l["label"].as_str().map(str::to_owned))
            .collect();

        let mut header = format!(
            "{}\n\nAllowed answers:\n",
            ds["instruction"].as_str().unwrap_or("").trim()
        );
        for l in &labels {
            header.push_str(&format!(
                "- {}: {}\n",
                l["label"].as_str().unwrap_or(""),
                l["definition"].as_str().unwrap_or("")
            ));
        }

        for case in ds["cases"].as_array().into_iter().flatten() {
            let Some(input) = case["input"].as_str() else {
                continue;
            };
            prompts.push(Prompt {
                dataset: name.clone(),
                req: DecisionRequest {
                    prompt: format!(
                        "{header}\nInput: {input}\n\nReply with exactly one allowed answer and nothing else."
                    ),
                    choices: choices.clone(),
                    temperature: None,
                },
            });
        }
    }
    if prompts.is_empty() {
        bail!("no prompts found in {dir}");
    }
    Ok(prompts)
}

fn main() -> Result<()> {
    let model_dir = env_or("KRONOS_MODEL_DIR", "./models/Qwen3-0.6B");
    let dataset_dir = env_or("KRONOS_BENCH_DATASETS", "eval/datasets");
    let iterations: usize = std::env::var("KRONOS_BENCH_ITERS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(10_000);

    // Prompts already list the labels, so Kronos's own choice appending is off.
    let config = KronosConfig {
        model_dir: PathBuf::from(&model_dir),
        append_choices_to_prompt: false,
        ..KronosConfig::default()
    };
    let kronos = EmbeddedKronos::from_config(&config)?;
    println!(
        "model={:?} device={} format={:?}",
        kronos.model_kind(),
        kronos.device_label(),
        kronos.prompt_format()
    );

    // Preflight: run every prompt once (untimed) and drop any that Kronos
    // rejects, e.g. a dataset whose labels are not single tokens for this model.
    let mut prompts = Vec::new();
    let mut rejected = 0usize;
    let mut reported = HashSet::new();
    for p in load_prompts(&dataset_dir)? {
        match kronos.evaluate(&p.req) {
            Ok(_) => prompts.push(p),
            Err(e) => {
                rejected += 1;
                if reported.insert(p.dataset.clone()) {
                    eprintln!("skipping rejected '{}' prompts: {e}", p.dataset);
                }
            }
        }
    }
    if prompts.is_empty() {
        bail!("every prompt was rejected; check the dataset labels for this model");
    }
    println!(
        "prompts: {} usable, {} rejected, from {}",
        prompts.len(),
        rejected,
        dataset_dir
    );

    for i in 0..WARMUP_PASSES {
        kronos.evaluate(&prompts[i % prompts.len()].req)?;
    }

    let mut samples = Vec::with_capacity(iterations);
    let mut by_dataset: BTreeMap<String, Vec<u64>> = BTreeMap::new();
    let started = Instant::now();
    for i in 0..iterations {
        let p = &prompts[i % prompts.len()];
        let t = Instant::now();
        kronos.evaluate(&p.req)?;
        let us = t.elapsed().as_micros() as u64;
        samples.push(us);
        by_dataset.entry(p.dataset.clone()).or_default().push(us);
        if (i + 1) % 1000 == 0 {
            eprintln!("  {}/{iterations}", i + 1);
        }
    }
    let total_secs = started.elapsed().as_secs_f64();
    samples.sort_unstable();

    let ms = |us: u64| us as f64 / 1000.0;
    let mean_ms = samples.iter().sum::<u64>() as f64 / samples.len().max(1) as f64 / 1000.0;

    println!("\n=== Kronos latency: {iterations} passes over {} prompts ===", prompts.len());
    println!("warmup:     {WARMUP_PASSES} passes (not recorded)");
    println!("mean:       {:>9.3} ms", mean_ms);
    println!("p50:        {:>9.3} ms", ms(percentile(&samples, 0.50)));
    println!("p95:        {:>9.3} ms", ms(percentile(&samples, 0.95)));
    println!("p99:        {:>9.3} ms", ms(percentile(&samples, 0.99)));
    println!("p99.9:      {:>9.3} ms", ms(percentile(&samples, 0.999)));
    println!("max:        {:>9.3} ms", ms(*samples.last().unwrap_or(&0)));
    println!("total time: {:>9.1} s", total_secs);
    println!("throughput: {:>9.2} decisions/s", iterations as f64 / total_secs);
    if iterations < 10_000 {
        println!("note: p99.9 rests on fewer than 10 samples at {iterations} passes; treat it as rough");
    }

    println!("\n=== By dataset ===");
    println!("{:<28} {:>7} {:>10} {:>10} {:>10} {:>10}", "dataset", "passes", "p50 ms", "p95 ms", "p99 ms", "max ms");
    for (name, mut v) in by_dataset {
        v.sort_unstable();
        println!(
            "{:<28} {:>7} {:>10.3} {:>10.3} {:>10.3} {:>10.3}",
            name,
            v.len(),
            ms(percentile(&v, 0.50)),
            ms(percentile(&v, 0.95)),
            ms(percentile(&v, 0.99)),
            ms(*v.last().unwrap_or(&0)),
        );
    }
    Ok(())
}
