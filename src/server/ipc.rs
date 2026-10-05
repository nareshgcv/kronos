//! Unix-socket IPC server: newline-delimited JSON.
//!
//! One JSON request per line, one JSON reply per line. Requests on the same
//! connection are answered in order. This is a local socket, not shared memory;
//! it avoids TCP/HTTP overhead but still serializes JSON.

use crate::engine::{Decision, DecisionRequest};
use crate::EmbeddedKronos;
use anyhow::{Context, Result};
use serde::Serialize;
use std::os::unix::fs::PermissionsExt;
use std::path::PathBuf;
use std::sync::Arc;
use tokio::io::{AsyncBufReadExt, AsyncReadExt, AsyncWriteExt, BufReader};
use tokio::net::unix::OwnedWriteHalf;
use tokio::net::{UnixListener, UnixStream};
use tokio::sync::Semaphore;

#[derive(Serialize)]
#[serde(tag = "status", rename_all = "lowercase")]
enum IpcReply {
    Ok { decision: Decision },
    Error { kind: &'static str, error: String },
}

impl IpcReply {
    fn error(kind: &'static str, error: impl Into<String>) -> Self {
        Self::Error {
            kind,
            error: error.into(),
        }
    }
}

pub struct IpcServer {
    socket_path: PathBuf,
    kronos: EmbeddedKronos,
    queue: Arc<Semaphore>,
    max_line_bytes: usize,
}

impl IpcServer {
    pub fn new(socket_path: PathBuf, kronos: EmbeddedKronos, queue: Arc<Semaphore>, max_line_bytes: usize) -> Self {
        Self {
            socket_path,
            kronos,
            queue,
            max_line_bytes,
        }
    }

    pub async fn run(self) -> Result<()> {
        if self.socket_path.exists() {
            std::fs::remove_file(&self.socket_path)
                .with_context(|| format!("removing stale socket {}", self.socket_path.display()))?;
        }
        let listener = UnixListener::bind(&self.socket_path)
            .with_context(|| format!("binding {}", self.socket_path.display()))?;
        // Owner-only: anything that can connect can drive the model.
        std::fs::set_permissions(&self.socket_path, std::fs::Permissions::from_mode(0o600))?;
        tracing::info!("IPC listening on {}", self.socket_path.display());

        let this = Arc::new(self);
        loop {
            let (stream, _) = listener.accept().await?;
            let this = Arc::clone(&this);
            tokio::spawn(async move {
                if let Err(e) = this.handle(stream).await {
                    tracing::debug!("IPC connection closed: {e:#}");
                }
            });
        }
    }

    async fn handle(&self, stream: UnixStream) -> Result<()> {
        let (reader, mut writer) = stream.into_split();
        let mut reader = BufReader::new(reader);
        let mut buf = Vec::with_capacity(4096);
        let limit = self.max_line_bytes as u64 + 1;

        loop {
            buf.clear();
            // Bounded read: a client can't make us buffer an unbounded line.
            let n = (&mut reader).take(limit).read_until(b'\n', &mut buf).await?;
            if n == 0 {
                return Ok(());
            }
            if buf.last() != Some(&b'\n') && n as u64 >= limit {
                let reply = IpcReply::error(
                    "invalid_request",
                    format!("request exceeds {} bytes", self.max_line_bytes),
                );
                write_reply(&mut writer, &reply).await?;
                return Ok(()); // framing is lost, close the connection
            }
            if buf.iter().all(u8::is_ascii_whitespace) {
                continue;
            }

            let reply = self.process(&buf).await;
            write_reply(&mut writer, &reply).await?;
        }
    }

    async fn process(&self, line: &[u8]) -> IpcReply {
        let req: DecisionRequest = match serde_json::from_slice(line) {
            Ok(r) => r,
            Err(e) => return IpcReply::error("invalid_request", format!("invalid JSON: {e}")),
        };
        let permit = match Arc::clone(&self.queue).try_acquire_owned() {
            Ok(p) => p,
            Err(_) => return IpcReply::error("busy", "server busy, retry later"),
        };
        let kronos = self.kronos.clone();

        match tokio::task::spawn_blocking(move || {
            let _permit = permit;
            kronos.evaluate(&req)
        })
        .await
        {
            Ok(Ok(decision)) => IpcReply::Ok { decision },
            Ok(Err(e)) => IpcReply::error(e.kind(), e.to_string()),
            Err(e) => IpcReply::error("inference_error", format!("worker task failed: {e}")),
        }
    }
}

async fn write_reply(writer: &mut OwnedWriteHalf, reply: &IpcReply) -> Result<()> {
    let mut bytes = serde_json::to_vec(reply)?;
    bytes.push(b'\n');
    writer.write_all(&bytes).await?;
    Ok(())
}
