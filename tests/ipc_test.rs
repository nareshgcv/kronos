//! Needs a running server:
//!   cargo run --release -- ./models/qwen2.5-0.5b-instruct
//!   cargo test --test ipc_test -- --ignored
#![cfg(unix)]

use serde_json::{json, Value};
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::net::UnixStream;

#[tokio::test]
#[ignore = "requires a running Kronos server"]
async fn ipc_roundtrip_and_error_handling() -> anyhow::Result<()> {
    let path = std::env::var("KRONOS_IPC_SOCKET").unwrap_or_else(|_| "/tmp/kronos.sock".to_owned());
    let stream = UnixStream::connect(&path).await?;
    let (reader, mut writer) = stream.into_split();
    let mut lines = BufReader::new(reader).lines();

    // 1. Valid request (newline-terminated!)
    let payload = json!({
        "prompt": "Player HP 12%, ammo 5%, 8 enemies within 2m. What should the game director do?",
        "choices": ["heal", "retreat", "attack"],
        "temperature": 0.8,
        "min_confidence": 0.5
    });
    let mut bytes = serde_json::to_vec(&payload)?;
    bytes.push(b'\n');
    writer.write_all(&bytes).await?;

    let line = lines.next_line().await?.expect("server closed connection");
    let reply: Value = serde_json::from_str(&line)?;
    assert_eq!(reply["status"], "ok", "reply: {reply}");
    let total: f64 = reply["decision"]["probabilities"]
        .as_array()
        .expect("probabilities array")
        .iter()
        .map(|c| c["probability"].as_f64().unwrap())
        .sum();
    assert!((total - 1.0).abs() < 1e-3);

    // 2. Malformed JSON -> error reply, connection stays open
    writer.write_all(b"{not json}\n").await?;
    let line = lines.next_line().await?.expect("server closed connection");
    let reply: Value = serde_json::from_str(&line)?;
    assert_eq!(reply["status"], "error");
    assert_eq!(reply["kind"], "invalid_request");

    // 3. Multi-token choice -> schema violation
    writer
        .write_all(b"{\"prompt\":\"x\",\"choices\":[\"SPAWN_HEALTH_PACK_NOW\",\"HOLD_POSITION_NOW\"]}\n")
        .await?;
    let line = lines.next_line().await?.expect("server closed connection");
    let reply: Value = serde_json::from_str(&line)?;
    assert_eq!(reply["kind"], "schema_violation");

    Ok(())
}
