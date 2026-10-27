//! Kronos evaluation harness: decision quality on labeled datasets.
//!
//!   cargo run --release --bin kronos-eval -- \
//!       --model ./models/Qwen3-0.6B \
//!       --dataset eval/datasets/routing.json \
//!       --style defined --order-check
//!
//! Add `--features cuda` or `--features metal` for GPU runs.
//! Uses Kronos only through its public embedded API, exactly as a user would.

use anyhow::{bail, Context, Result};
use kronos::utils::eval_metrics::{
    accuracy, calibration, confusion_matrix, macro_f1, mean_confidence_split, per_class,
    threshold_sweep, Outcome,
};
use kronos::utils::metrics::percentile;
use kronos::{DecisionRequest, EmbeddedKronos, KronosConfig};
use serde::Deserialize;
use std::collections::HashSet;
use std::fmt::Write as _;
use std::path::{Path, PathBuf};
use std::time::{Instant, SystemTime, UNIX_EPOCH};

const THRESHOLDS: [f64; 7] = [0.0, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95];
const CALIBRATION_BINS: usize = 10;
const WARMUP_RUNS: usize = 3;

// ---------------------------------------------------------------- dataset

#[derive(Debug, Deserialize)]
struct Dataset {
    task: String,
    instruction: String,
    labels: Vec<LabelDef>,
    #[serde(default)]
    examples: Vec<Example>,
    cases: Vec<Case>,
}

#[derive(Debug, Deserialize)]
struct LabelDef {
    label: String,
    definition: String,
}

#[derive(Debug, Deserialize)]
struct Example {
    input: String,
    label: String,
}

#[derive(Debug, Deserialize)]
struct Case {
    id: String,
    input: String,
    label: String,
    #[serde(default = "default_split")]
    split: String,
    #[serde(default)]
    tags: Vec<String>,
}

fn default_split() -> String {
    "test".to_owned()
}

fn load_dataset(path: &Path) -> Result<Dataset> {
    let text = std::fs::read_to_string(path)
        .with_context(|| format!("cannot read dataset {}", path.display()))?;
    let ds: Dataset = serde_json::from_str(&text)
        .with_context(|| format!("invalid dataset JSON in {}", path.display()))?;

    if ds.labels.len() < 2 {
        bail!("{}: needs at least 2 labels", path.display());
    }
    let mut seen = HashSet::new();
    for l in &ds.labels {
        if !seen.insert(l.label.as_str()) {
            bail!("{}: duplicate label '{}'", path.display(), l.label);
        }
    }
    for e in &ds.examples {
        if !seen.contains(e.label.as_str()) {
            bail!("{}: example uses unknown label '{}'", path.display(), e.label);
        }
    }
    let mut ids = HashSet::new();
    for c in &ds.cases {
        if !seen.contains(c.label.as_str()) {
            bail!("{}: case {} uses unknown label '{}'", path.display(), c.id, c.label);
        }
        if !ids.insert(c.id.as_str()) {
            bail!("{}: duplicate case id '{}'", path.display(), c.id);
        }
        if c.split != "test" && c.split != "calibration" {
            bail!("{}: case {} has split '{}' (use test or calibration)", path.display(), c.id, c.split);
        }
    }
    Ok(ds)
}

// ---------------------------------------------------------------- arguments

#[derive(Debug, Clone, Copy, PartialEq)]
enum Style {
    /// Labels only.
    Bare,
    /// Labels with one-line definitions.
    Defined,
    /// Definitions plus the dataset's worked examples.
    FewShot,
}

impl Style {
    fn parse(s: &str) -> Result<Self> {
        Ok(match s {
            "bare" => Style::Bare,
            "defined" => Style::Defined,
            "fewshot" => Style::FewShot,
            other => bail!("unknown --style '{other}' (use bare, defined or fewshot)"),
        })
    }
    fn as_str(self) -> &'static str {
        match self {
            Style::Bare => "bare",
            Style::Defined => "defined",
            Style::FewShot => "fewshot",
        }
    }
}

#[derive(Debug, Clone, Copy)]
enum Split {
    Test,
    Calibration,
    All,
}

impl Split {
    fn parse(s: &str) -> Result<Self> {
        Ok(match s {
            "test" => Split::Test,
            "calibration" => Split::Calibration,
            "all" => Split::All,
            other => bail!("unknown --split '{other}' (use test, calibration or all)"),
        })
    }
    fn as_str(self) -> &'static str {
        match self {
            Split::Test => "test",
            Split::Calibration => "calibration",
            Split::All => "all",
        }
    }
    fn includes(self, split: &str) -> bool {
        match self {
            Split::All => true,
            other => other.as_str() == split,
        }
    }
}

struct Args {
    model: String,
    datasets: Vec<PathBuf>,
    style: Style,
    split: Split,
    order_check: bool,
    out: PathBuf,
}

const USAGE: &str = "\
kronos-eval --model DIR --dataset FILE [--dataset FILE ...] [options]

Options:
  --style bare|defined|fewshot     prompt style (default: defined)
  --split test|calibration|all     which cases to run (default: test)
  --order-check                    rerun with choices reversed and report flips
  --out DIR                        report directory (default: eval/results)

The model directory defaults to $KRONOS_MODEL_DIR.";

fn parse_args() -> Result<Args> {
    let mut args = Args {
        model: std::env::var("KRONOS_MODEL_DIR").unwrap_or_default(),
        datasets: Vec::new(),
        style: Style::Defined,
        split: Split::Test,
        order_check: false,
        out: PathBuf::from("eval/results"),
    };
    let mut it = std::env::args().skip(1);
    while let Some(flag) = it.next() {
        let mut value = || it.next().with_context(|| format!("{flag} needs a value"));
        match flag.as_str() {
            "--model" => args.model = value()?,
            "--dataset" => args.datasets.push(PathBuf::from(value()?)),
            "--style" => args.style = Style::parse(&value()?)?,
            "--split" => args.split = Split::parse(&value()?)?,
            "--out" => args.out = PathBuf::from(value()?),
            "--order-check" => args.order_check = true,
            "-h" | "--help" => {
                println!("{USAGE}");
                std::process::exit(0);
            }
            other => bail!("unknown argument '{other}'\n\n{USAGE}"),
        }
    }
    if args.model.is_empty() {
        bail!("no model directory: pass --model or set KRONOS_MODEL_DIR\n\n{USAGE}");
    }
    if args.datasets.is_empty() {
        bail!("no dataset: pass at least one --dataset\n\n{USAGE}");
    }
    Ok(args)
}

// ---------------------------------------------------------------- running

fn build_prompt(ds: &Dataset, choices: &[String], style: Style, input: &str) -> String {
    let mut p = String::new();
    let _ = writeln!(p, "{}", ds.instruction.trim());
    let _ = writeln!(p);

    if style == Style::Bare {
        let _ = writeln!(p, "Allowed answers: {}", choices.join(", "));
    } else {
        let _ = writeln!(p, "Allowed answers:");
        for c in choices {
            let def = ds
                .labels
                .iter()
                .find(|l| &l.label == c)
                .map(|l| l.definition.as_str())
                .unwrap_or("");
            let _ = writeln!(p, "- {c}: {def}");
        }
    }

    if style == Style::FewShot && !ds.examples.is_empty() {
        let _ = writeln!(p, "\nExamples:");
        for e in &ds.examples {
            let _ = writeln!(p, "Input: {}\nAnswer: {}\n", e.input, e.label);
        }
    }

    let _ = writeln!(p, "\nInput: {input}\n");
    let _ = write!(p, "Reply with exactly one allowed answer and nothing else.");
    p
}

struct CaseResult {
    id: String,
    gold: String,
    tags: Vec<String>,
    predicted: Option<String>,
    confidence: f64,
    latency_us: u64,
    error: Option<String>,
}

impl CaseResult {
    fn outcome(&self) -> Outcome {
        Outcome {
            gold: self.gold.clone(),
            predicted: self.predicted.clone(),
            confidence: self.confidence,
        }
    }
}

fn request(ds: &Dataset, choices: &[String], style: Style, input: &str) -> DecisionRequest {
    DecisionRequest {
        prompt: build_prompt(ds, choices, style, input),
        choices: choices.to_vec(),
        temperature: None,
    }
}

fn run_pass(
    kronos: &EmbeddedKronos,
    ds: &Dataset,
    cases: &[&Case],
    choices: &[String],
    style: Style,
) -> Vec<CaseResult> {
    let mut out = Vec::with_capacity(cases.len());
    for (i, case) in cases.iter().enumerate() {
        let req = request(ds, choices, style, &case.input);
        let started = Instant::now();
        let result = kronos.evaluate(&req);
        let latency_us = started.elapsed().as_micros() as u64;

        let base = CaseResult {
            id: case.id.clone(),
            gold: case.label.clone(),
            tags: case.tags.clone(),
            predicted: None,
            confidence: 0.0,
            latency_us,
            error: None,
        };
        out.push(match result {
            Ok(d) => CaseResult {
                predicted: Some(d.selected_choice.to_string()),
                confidence: d.confidence as f64,
                ..base
            },
            Err(e) => CaseResult {
                error: Some(e.to_string()),
                ..base
            },
        });
        if (i + 1) % 10 == 0 {
            eprintln!("  {}/{}", i + 1, cases.len());
        }
    }
    out
}

// ---------------------------------------------------------------- report

struct RunMeta {
    model_dir: String,
    model_kind: String,
    device: String,
    prompt_format: String,
    commit: String,
    timestamp: u64,
}

fn git_commit() -> String {
    std::process::Command::new("git")
        .args(["rev-parse", "--short", "HEAD"])
        .output()
        .ok()
        .filter(|o| o.status.success())
        .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_owned())
        .unwrap_or_else(|| "unknown".to_owned())
}

fn pct(x: f64) -> String {
    format!("{:.1}%", x * 100.0)
}

fn opt3(x: Option<f64>) -> String {
    x.map(|v| format!("{v:.3}")).unwrap_or_else(|| "n/a".to_owned())
}

fn render_report(
    meta: &RunMeta,
    args: &Args,
    ds: &Dataset,
    path: &Path,
    labels: &[String],
    results: &[CaseResult],
    reversed: Option<&[CaseResult]>,
) -> String {
    let outcomes: Vec<Outcome> = results.iter().map(CaseResult::outcome).collect();
    let cm = confusion_matrix(labels, &outcomes);
    let stats = per_class(&cm);
    let (bins, ece) = calibration(&outcomes, CALIBRATION_BINS);
    let sweep = threshold_sweep(&outcomes, &THRESHOLDS);
    let (conf_right, conf_wrong) = mean_confidence_split(&outcomes);

    let errors: Vec<&CaseResult> = results.iter().filter(|r| r.error.is_some()).collect();
    let mut lat: Vec<u64> = results.iter().map(|r| r.latency_us).collect();
    lat.sort_unstable();
    let ms = |us: u64| us as f64 / 1000.0;

    let mut r = String::new();
    let _ = writeln!(r, "# Kronos evaluation: {}\n", ds.task);
    let _ = writeln!(r, "| Setting | Value |\n|---|---|");
    let _ = writeln!(r, "| Dataset | `{}` |", path.display());
    let _ = writeln!(r, "| Split | {} |", args.split.as_str());
    let _ = writeln!(r, "| Prompt style | {} |", args.style.as_str());
    let _ = writeln!(r, "| Model dir | `{}` |", meta.model_dir);
    let _ = writeln!(r, "| Model kind | {} |", meta.model_kind);
    let _ = writeln!(r, "| Prompt format | {} |", meta.prompt_format);
    let _ = writeln!(r, "| Device | {} |", meta.device);
    let _ = writeln!(r, "| Git commit | {} |", meta.commit);
    let _ = writeln!(r, "| Unix time | {} |\n", meta.timestamp);

    let _ = writeln!(r, "## Summary\n");
    let _ = writeln!(r, "| Metric | Value |\n|---|---|");
    let _ = writeln!(r, "| Cases | {} |", results.len());
    let _ = writeln!(r, "| Errors (no decision) | {} |", errors.len());
    let _ = writeln!(r, "| Accuracy | {} |", pct(accuracy(&outcomes)));
    let _ = writeln!(r, "| Macro F1 | {:.3} |", macro_f1(&stats));
    let _ = writeln!(r, "| Mean confidence, correct | {} |", opt3(conf_right));
    let _ = writeln!(r, "| Mean confidence, wrong | {} |", opt3(conf_wrong));
    let _ = writeln!(r, "| ECE ({CALIBRATION_BINS} bins) | {ece:.3} |");
    let _ = writeln!(
        r,
        "| Latency p50 / p95 / p99 / max | {:.1} / {:.1} / {:.1} / {:.1} ms |\n",
        ms(percentile(&lat, 0.50)),
        ms(percentile(&lat, 0.95)),
        ms(percentile(&lat, 0.99)),
        ms(*lat.last().unwrap_or(&0)),
    );
    if let Some(first) = errors.first() {
        let _ = writeln!(
            r,
            "> First error ({}): {}\n",
            first.id,
            first.error.as_deref().unwrap_or("")
        );
    }

    let _ = writeln!(r, "## Per-class results\n");
    let _ = writeln!(r, "| Label | Precision | Recall | F1 | Support |\n|---|---|---|---|---|");
    for s in &stats {
        let _ = writeln!(
            r,
            "| {} | {:.3} | {:.3} | {:.3} | {} |",
            s.label, s.precision, s.recall, s.f1, s.support
        );
    }

    let _ = writeln!(r, "\n## Confusion matrix (rows = correct label, columns = Kronos)\n");
    let _ = write!(r, "| |");
    for l in labels {
        let _ = write!(r, " {l} |");
    }
    let _ = writeln!(r, " error |");
    let _ = writeln!(r, "|---|{}---|", "---|".repeat(labels.len()));
    for (i, l) in labels.iter().enumerate() {
        let _ = write!(r, "| **{l}** |");
        for c in &cm.counts[i] {
            let _ = write!(r, " {c} |");
        }
        let _ = writeln!(r, " {} |", cm.errors[i]);
    }

    let _ = writeln!(r, "\n## Calibration\n");
    let _ = writeln!(r, "| Confidence bin | Cases | Avg confidence | Accuracy |\n|---|---|---|---|");
    for b in bins.iter().filter(|b| b.count > 0) {
        let _ = writeln!(
            r,
            "| {:.1}–{:.1} | {} | {:.3} | {} |",
            b.lower,
            b.upper,
            b.count,
            b.avg_confidence,
            pct(b.accuracy)
        );
    }

    let _ = writeln!(r, "\n## Abstention thresholds\n");
    let _ = writeln!(
        r,
        "Abstain when confidence is below the threshold.\n\n\
         | Threshold | Coverage | Accuracy when answering | Wrong answers caught | Correct answers lost |\n\
         |---|---|---|---|---|"
    );
    for row in &sweep {
        let _ = writeln!(
            r,
            "| {:.2} | {} | {} | {} | {} |",
            row.threshold,
            pct(row.coverage),
            pct(row.selective_accuracy),
            pct(row.wrong_caught),
            pct(row.correct_lost)
        );
    }

    if let Some(rev) = reversed {
        let rev_outcomes: Vec<Outcome> = rev.iter().map(CaseResult::outcome).collect();
        let flips: Vec<(&CaseResult, &CaseResult)> = results
            .iter()
            .zip(rev.iter())
            .filter(|(a, b)| a.predicted != b.predicted)
            .collect();
        let _ = writeln!(r, "\n## Choice-order check\n");
        let _ = writeln!(r, "| Metric | Value |\n|---|---|");
        let _ = writeln!(r, "| Accuracy, original order | {} |", pct(accuracy(&outcomes)));
        let _ = writeln!(r, "| Accuracy, reversed order | {} |", pct(accuracy(&rev_outcomes)));
        let _ = writeln!(
            r,
            "| Answers that changed | {} of {} ({}) |",
            flips.len(),
            results.len(),
            pct(flips.len() as f64 / results.len().max(1) as f64)
        );
        for (a, b) in &flips {
            let _ = writeln!(
                r,
                "\n- `{}` (correct: {}): {} → {}",
                a.id,
                a.gold,
                a.predicted.as_deref().unwrap_or("error"),
                b.predicted.as_deref().unwrap_or("error")
            );
        }
    }

    let mut mistakes: Vec<&CaseResult> = results
        .iter()
        .filter(|c| c.predicted.is_some() && c.predicted.as_deref() != Some(c.gold.as_str()))
        .collect();
    mistakes.sort_by(|a, b| b.confidence.total_cmp(&a.confidence));
    let _ = writeln!(r, "\n## Wrong answers (highest confidence first)\n");
    if mistakes.is_empty() {
        let _ = writeln!(r, "None.");
    } else {
        let _ = writeln!(
            r,
            "| Case | Correct | Kronos | Confidence | Tags |\n|---|---|---|---|---|"
        );
        for m in mistakes {
            let _ = writeln!(
                r,
                "| {} | {} | {} | {:.3} | {} |",
                m.id,
                m.gold,
                m.predicted.as_deref().unwrap_or(""),
                m.confidence,
                m.tags.join(", ")
            );
        }
    }

    let _ = writeln!(
        r,
        "\n---\nConfidence is relative to the allowed choices only, so it is not a \
         calibrated probability of being right. Check the calibration table before \
         using it as a threshold."
    );
    r
}

fn write_predictions(path: &Path, results: &[CaseResult]) -> Result<()> {
    let mut lines = String::new();
    for c in results {
        let row = serde_json::json!({
            "id": c.id,
            "gold": c.gold,
            "predicted": c.predicted,
            "correct": c.predicted.as_deref() == Some(c.gold.as_str()),
            "confidence": c.confidence,
            "latency_us": c.latency_us,
            "error": c.error,
            "tags": c.tags,
        });
        lines.push_str(&row.to_string());
        lines.push('\n');
    }
    std::fs::write(path, lines).with_context(|| format!("cannot write {}", path.display()))
}

// ---------------------------------------------------------------- main

fn main() -> Result<()> {
    let args = parse_args()?;
    // The harness writes the label list into every prompt itself, so Kronos's own
    // "append choices to prompt" option is turned off to avoid listing them twice.
    let config = KronosConfig {
        model_dir: PathBuf::from(&args.model),
        append_choices_to_prompt: false,
        ..KronosConfig::default()
    };
    let kronos = EmbeddedKronos::from_config(&config)?;
    let meta = RunMeta {
        model_dir: args.model.clone(),
        model_kind: format!("{:?}", kronos.model_kind()),
        device: kronos.device_label().to_string(),
        prompt_format: format!("{:?}", kronos.prompt_format()),
        commit: git_commit(),
        timestamp: SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0),
    };
    std::fs::create_dir_all(&args.out)
        .with_context(|| format!("cannot create {}", args.out.display()))?;
    eprintln!("model={} device={} format={}", meta.model_kind, meta.device, meta.prompt_format);

    for path in &args.datasets {
        let ds = load_dataset(path)?;
        let cases: Vec<&Case> = ds.cases.iter().filter(|c| args.split.includes(&c.split)).collect();
        if cases.is_empty() {
            eprintln!("[{}] no '{}' cases, skipping", ds.task, args.split.as_str());
            continue;
        }
        let labels: Vec<String> = ds.labels.iter().map(|l| l.label.clone()).collect();

        // Warmup also surfaces label problems (e.g. multi-token labels) before the timed run.
        let warm = request(&ds, &labels, args.style, &cases[0].input);
        for _ in 0..WARMUP_RUNS {
            if let Err(e) = kronos.evaluate(&warm) {
                eprintln!("[{}] warmup failed: {e}", ds.task);
                break;
            }
        }

        eprintln!("[{}] {} cases, style={}", ds.task, cases.len(), args.style.as_str());
        let forward = run_pass(&kronos, &ds, &cases, &labels, args.style);
        let reversed = if args.order_check {
            eprintln!("[{}] order check (choices reversed)", ds.task);
            let rev: Vec<String> = labels.iter().rev().cloned().collect();
            Some(run_pass(&kronos, &ds, &cases, &rev, args.style))
        } else {
            None
        };

        let report = render_report(&meta, &args, &ds, path, &labels, &forward, reversed.as_deref());
        println!("{report}");

        let stem = format!(
            "{}-{}-{}-{}",
            meta.timestamp,
            ds.task,
            args.style.as_str(),
            args.split.as_str()
        );
        let report_path = args.out.join(format!("{stem}.md"));
        std::fs::write(&report_path, &report)
            .with_context(|| format!("cannot write {}", report_path.display()))?;
        write_predictions(&args.out.join(format!("{stem}.jsonl")), &forward)?;
        eprintln!("[{}] report: {}", ds.task, report_path.display());
    }
    Ok(())
}
