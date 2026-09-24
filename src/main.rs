mod config;
mod engine;
mod server;
mod tokenizer;

use anyhow::Result;
use config::{select_best_device, KronosConfig};
use engine::candle_backend::CandleEngine;
use server::http::{create_router, AppState};
use server::ipc::IpcServer;
use std::net::SocketAddr;
use std::sync::Arc;

#[tokio::main]
async fn main() -> Result<()> {
    println!("==================================================");
    println!("     KRONOS CORE :: System 1 Engine Initializing   ");
    println!("==================================================");

    let config = KronosConfig::default();
    let device = select_best_device();

    println!("[Device] Selected Hardware Acceleration: {:?}", device);

    // 1. Load model weights into memory
    let engine = CandleEngine::load(&config.model_dir, device)?;
    let shared_engine = Arc::new(engine);

    // 2. Spawn Sub-2ms Unix IPC Socket Listener
    if config.use_ipc {
        if std::path::Path::new(&config.ipc_socket_path).exists() {
            let _ = std::fs::remove_file(&config.ipc_socket_path);
        }

        let ipc_server = IpcServer::new(config.ipc_socket_path.clone(), Arc::clone(&shared_engine));
        tokio::spawn(async move {
            if let Err(e) = ipc_server.run().await {
                eprintln!("[IPC Fatal] Server crashed: {}", e);
            }
        });
    }

    // 3. Start HTTP/REST API Server (Axum)
    let state = Arc::new(AppState {
        engine: (*shared_engine).clone_engine(),
    });

    let app = create_router(state);
    let addr = SocketAddr::from(([127, 0, 0, 1], config.server_port));

    println!("[HTTP] REST API listening on http://{}", addr);
    let listener = tokio::net::TcpListener::bind(addr).await?;

    axum::serve(listener, app)
        .with_graceful_shutdown(shutdown_signal())
        .await?;

    Ok(())
}

async fn shutdown_signal() {
    tokio::signal::ctrl_c()
        .await
        .expect("Failed to install Ctrl+C signal handler");
    println!("\n[System] Gracefully shutting down Kronos Core...");
}
