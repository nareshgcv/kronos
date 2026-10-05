//! Network layer: HTTP (all platforms) and Unix-socket IPC (unix only).

pub mod http;
#[cfg(unix)]
pub mod ipc;
