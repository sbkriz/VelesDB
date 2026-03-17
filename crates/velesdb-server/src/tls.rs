//! TLS configuration and server support via rustls.
//!
//! When TLS cert and key paths are provided, the server binds with HTTPS.
//! Otherwise, plain HTTP is used (default for local dev).

use std::io::BufReader;
use std::sync::Arc;

use rustls::ServerConfig as TlsServerConfig;
use tokio_rustls::TlsAcceptor;

/// Load TLS configuration from PEM certificate and private key files.
///
/// Returns a [`TlsAcceptor`] ready for use with `tokio` TCP streams.
///
/// # Errors
///
/// Returns an error if the certificate or key files cannot be read,
/// contain no valid PEM items, or if rustls rejects the configuration.
pub fn load_tls_config(cert_path: &str, key_path: &str) -> anyhow::Result<TlsAcceptor> {
    let cert_file = std::fs::File::open(cert_path)
        .map_err(|e| anyhow::anyhow!("failed to open TLS cert file '{cert_path}': {e}"))?;
    let key_file = std::fs::File::open(key_path)
        .map_err(|e| anyhow::anyhow!("failed to open TLS key file '{key_path}': {e}"))?;

    let certs: Vec<_> = rustls_pemfile::certs(&mut BufReader::new(cert_file))
        .collect::<Result<Vec<_>, _>>()
        .map_err(|e| anyhow::anyhow!("failed to parse TLS certificates from '{cert_path}': {e}"))?;

    if certs.is_empty() {
        anyhow::bail!("no certificates found in '{cert_path}'");
    }

    let key = rustls_pemfile::private_key(&mut BufReader::new(key_file))
        .map_err(|e| anyhow::anyhow!("failed to parse TLS private key from '{key_path}': {e}"))?
        .ok_or_else(|| anyhow::anyhow!("no private key found in '{key_path}'"))?;

    let config = TlsServerConfig::builder()
        .with_no_client_auth()
        .with_single_cert(certs, key)
        .map_err(|e| anyhow::anyhow!("invalid TLS configuration: {e}"))?;

    Ok(TlsAcceptor::from(Arc::new(config)))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_load_tls_config_missing_cert_file() {
        match load_tls_config("/nonexistent/cert.pem", "/nonexistent/key.pem") {
            Err(e) => assert!(e.to_string().contains("cert")),
            Ok(_) => panic!("should fail for missing files"),
        }
    }

    #[test]
    fn test_load_tls_config_empty_cert_file() {
        let dir = tempfile::tempdir().unwrap();
        let cert_path = dir.path().join("cert.pem");
        let key_path = dir.path().join("key.pem");
        std::fs::write(&cert_path, "").unwrap();
        std::fs::write(&key_path, "").unwrap();

        match load_tls_config(&cert_path.to_string_lossy(), &key_path.to_string_lossy()) {
            Err(e) => assert!(e.to_string().contains("no certificates")),
            Ok(_) => panic!("should fail for empty cert"),
        }
    }

    #[test]
    fn test_load_tls_config_invalid_pem() {
        let dir = tempfile::tempdir().unwrap();
        let cert_path = dir.path().join("cert.pem");
        let key_path = dir.path().join("key.pem");
        std::fs::write(&cert_path, "not a real cert").unwrap();
        std::fs::write(&key_path, "not a real key").unwrap();

        assert!(
            load_tls_config(&cert_path.to_string_lossy(), &key_path.to_string_lossy(),).is_err(),
            "should fail for invalid PEM"
        );
    }
}
