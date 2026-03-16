//! Tests for `RateLimiter` and `CircuitBreaker` resilience primitives.

use super::resilience::{CircuitBreaker, CircuitState, RateLimiter};
use crate::guardrails::limits::GuardRailViolation;

// ─────────────────────────────────────────────────────────────────────────────
// RateLimiter
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn rate_limiter_allows_within_limit() {
    let limiter = RateLimiter::new(10);
    // First 10 requests should succeed (initial bucket = limit)
    for i in 0..10 {
        assert!(
            limiter.check("client-a").is_ok(),
            "request {i} should be allowed"
        );
    }
}

#[test]
fn rate_limiter_rejects_over_limit() {
    let limiter = RateLimiter::new(3);
    // Exhaust all 3 tokens
    for _ in 0..3 {
        limiter.check("client-b").expect("should be allowed");
    }
    // 4th request should be rejected
    let result = limiter.check("client-b");
    assert!(result.is_err());
    match result.unwrap_err() {
        GuardRailViolation::RateLimitExceeded { limit_qps } => {
            assert_eq!(limit_qps, 3);
        }
        other => panic!("expected RateLimitExceeded, got: {other}"),
    }
}

#[test]
fn rate_limiter_isolates_clients() {
    let limiter = RateLimiter::new(2);
    // Exhaust client-a
    limiter.check("client-a").unwrap();
    limiter.check("client-a").unwrap();
    assert!(limiter.check("client-a").is_err());

    // client-b should still have tokens
    assert!(limiter.check("client-b").is_ok());
}

// ─────────────────────────────────────────────────────────────────────────────
// CircuitBreaker: state transitions
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn circuit_starts_closed() {
    let cb = CircuitBreaker::new(3, 60);
    assert_eq!(cb.state(), CircuitState::Closed);
    assert!(cb.check().is_ok());
}

#[test]
fn circuit_opens_after_failure_threshold() {
    let cb = CircuitBreaker::new(3, 60);

    cb.record_failure();
    assert_eq!(cb.state(), CircuitState::Closed);
    cb.record_failure();
    assert_eq!(cb.state(), CircuitState::Closed);
    cb.record_failure();
    // 3 failures with threshold=3 should open
    assert_eq!(cb.state(), CircuitState::Open);
}

#[test]
fn circuit_open_rejects_requests() {
    let cb = CircuitBreaker::new(1, 9999);
    cb.record_failure(); // threshold=1 => opens immediately
    assert_eq!(cb.state(), CircuitState::Open);

    let result = cb.check();
    assert!(result.is_err());
    match result.unwrap_err() {
        GuardRailViolation::CircuitOpen {
            recovery_in_seconds,
        } => {
            assert!(recovery_in_seconds > 0);
        }
        other => panic!("expected CircuitOpen, got: {other}"),
    }
}

#[test]
fn circuit_success_resets_failure_count() {
    let cb = CircuitBreaker::new(3, 60);
    cb.record_failure();
    cb.record_failure();
    // 2 failures, then success resets
    cb.record_success();
    assert_eq!(cb.state(), CircuitState::Closed);

    // Should need 3 more failures to open
    cb.record_failure();
    cb.record_failure();
    assert_eq!(cb.state(), CircuitState::Closed);
    cb.record_failure();
    assert_eq!(cb.state(), CircuitState::Open);
}

#[test]
fn circuit_halfopen_success_closes() {
    let cb = CircuitBreaker::new(1, 0); // recovery_seconds=0 => immediate
    cb.record_failure();
    assert_eq!(cb.state(), CircuitState::Open);

    // With recovery_seconds=0, check() transitions to HalfOpen
    assert!(cb.check().is_ok());
    assert_eq!(cb.state(), CircuitState::HalfOpen);

    // A success in HalfOpen should close the circuit
    cb.record_success();
    assert_eq!(cb.state(), CircuitState::Closed);
}

#[test]
fn circuit_halfopen_failure_reopens() {
    let cb = CircuitBreaker::new(1, 0);
    cb.record_failure();
    assert_eq!(cb.state(), CircuitState::Open);

    // Transition to HalfOpen
    assert!(cb.check().is_ok());
    assert_eq!(cb.state(), CircuitState::HalfOpen);

    // Failure in HalfOpen should reopen
    cb.record_failure();
    assert_eq!(cb.state(), CircuitState::Open);
}
