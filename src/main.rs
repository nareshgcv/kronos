use anyhow::Result;
use kronos::server::http::{create_router, AppState};
use kronos::{EmbeddedKronos, KronosConfig};
use std::sync::Arc;
use tokio::sync::Semaphore;
use tracing::{error, info};
use tracing_subscriber::EnvFilter;

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info")))
        .init();

    // Config: defaults < KRONOS_* env vars < first CLI arg (model dir)
    let mut config = KronosConfig::from_env()?;
    if let Some(dir) = std::env::args().nth(1) {
        config.model_dir = dir.into();
    }
    config.validate()?;

    info!("loading model from {}", config.model_dir.display());
    let kronos = {
        let cfg = config.clone();
        tokio::task::spawn_blocking(move || EmbeddedKronos::from_config(&cfg)).await??
    };
    info!(
        model = ?kronos.model_kind(),
        device = kronos.device_label(),
        prompt_format = ?kronos.prompt_format(),
        "model loaded"
    );

    if config.warmup {
        let k = kronos.clone();
        tokio::task::spawn_blocking(move || k.warmup()).await??;
        info!("warmup pass complete");
    }

    let queue = Arc::new(Semaphore::new(config.max_queued_requests));

    #[cfg(unix)]
    {
        if config.use_ipc {
            let ipc = kronos::server::ipc::IpcServer::new(
                config.ipc_socket_path.clone(),
                kronos.clone(),
                Arc::clone(&queue),
                config.max_request_bytes,
            );
            tokio::spawn(async move {
                if let Err(e) = ipc.run().await {
                    error!("IPC server stopped: {e:#}");
                }
            });
        }
    }

    let app = create_router(AppState { kronos, queue }, config.max_request_bytes);
    let listener = tokio::net::TcpListener::bind(config.http_addr).await?;
    info!("HTTP listening on http://{}", config.http_addr);

    axum::serve(listener, app)
        .with_graceful_shutdown(shutdown_signal())
        .await?;

    #[cfg(unix)]
    {
        if config.use_ipc {
            let _ = std::fs::remove_file(&config.ipc_socket_path);
        }
    }
    info!("shut down cleanly");
    Ok(())
}

async fn shutdown_signal() {
    let ctrl_c = async {
        let _ = tokio::signal::ctrl_c().await;
    };

    #[cfg(unix)]
    let terminate = async {
        use tokio::signal::unix::{signal, SignalKind};
        match signal(SignalKind::terminate()) {
            Ok(mut s) => {
                s.recv().await;
            }
            Err(_) => std::future::pending::<()>().await,
        }
    };
    #[cfg(not(unix))]
    let terminate = std::future::pending::<()>();

    tokio::select! {
        _ = ctrl_c => {},
        _ = terminate => {},
    }
    info!("shutdown signal received");
}
