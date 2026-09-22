use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::time::Instant;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::UnixStream;

#[derive(Debug, Serialize, Deserialize)]
struct IpcResponse {
    selected_choice: String,
    confidence: f32,
    latency_ms: f64,
}

#[tokio::main]
async fn main() -> Result<()> {
    let socket_path = "/tmp/kronos.sock";

    println!("==================================================");
    println!("     KRONOS CORE :: Sub-10ms IPC Test Suite       ");
    println!("==================================================");

    // 1. Establish connection to local Kronos Unix Socket
    println!("[IPC Client] Connecting to socket at: {}", socket_path);
    let mut stream = UnixStream::connect(socket_path)
        .await
        .with_context(|| format!("Failed to connect to Kronos IPC socket at '{}'. Is Kronos server running?", socket_path))?;

    println!("[IPC Client] Connection established. Sending telemetry state...");

    // 2. Construct sample real-time game telemetry payload
    let payload = json!({
        "prompt": "SYS_STATE: PLAYER_HP=12% AMMO=5% ENEMIES_NEARBY=8 DISTANCE=2m | ACTION_DECISION:",
        "choices": [
            "SPAWN_HEALTH",
            "SPAWN_AMMO",
            "TRIGGER_FLANK",
            "HOLD"
        ],
        "temperature": 0.8
    });

    let payload_bytes = serde_json::to_vec(&payload)?;

    // 3. Measure total client-side roundtrip latency
    let start_time = Instant::now();

    // Send payload packet
    stream.write_all(&payload_bytes).await?;

    // Read response packet
    let mut buffer = vec![0u8; 4096];
    let bytes_read = stream.read(&mut buffer).await?;

    let total_roundtrip_ms = start_time.elapsed().as_secs_f64() * 1000.0;

    if bytes_read == 0 {
        anyhow::bail!("Server closed connection without responding.");
    }

    // 4. Deserialize and display response
    let response: IpcResponse = serde_json::from_slice(&buffer[..bytes_read])
        .context("Failed to deserialize response from Kronos server.")?;

    println!("\n==================================================");
    println!("              KRONOS DECISION RESULT              ");
    println!("==================================================");
    println!("Selected Decision       : {}", response.selected_choice);
    println!("Probability Confidence  : {:.2}%", response.confidence * 100.0);
    println!("Server Inference Time   : {:.2} ms", response.latency_ms);
    println!("Total IPC Roundtrip Time: {:.2} ms", total_roundtrip_ms);
    println!("==================================================");

    if total_roundtrip_ms < 10.0 {
        println!("✅ SUCCESS: Decision delivered under 10ms SLA!");
    } else {
        println!("⚠️ WARNING: Latency exceeded 10ms threshold.");
    }

    Ok(())
      }
