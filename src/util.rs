//! Shared utility functions used across modules.

/// Acquire a mutex guard, recovering from a poisoned mutex rather than
/// propagating an error.  A panicking thread does not permanently break a
/// shared resource; we simply take ownership of the inner value and log a
/// warning so that contention hot-spots can be identified.
pub fn recover_lock<'a, T>(
    result: std::sync::LockResult<std::sync::MutexGuard<'a, T>>,
    ctx: &str,
) -> std::sync::MutexGuard<'a, T>
where
    T: ?Sized,
{
    match result {
        Ok(guard) => guard,
        Err(poisoned) => {
            tracing::warn!("mutex poisoned in {ctx}, recovering inner value");
            poisoned.into_inner()
        }
    }
}

/// Acquire a mutex guard with timing and poison recovery.
///
/// Logs a warning if acquisition takes > 5 ms (contention hot-spot indicator).
/// Recovers from a poisoned mutex rather than propagating the error.
pub fn timed_lock<'a, T>(mutex: &'a std::sync::Mutex<T>, ctx: &str) -> std::sync::MutexGuard<'a, T>
where
    T: ?Sized,
{
    let start = std::time::Instant::now();
    let result = mutex.lock();
    let elapsed = start.elapsed();
    if elapsed > std::time::Duration::from_millis(5) {
        tracing::warn!(
            duration_ms = elapsed.as_millis(),
            ctx = ctx,
            "slow mutex acquisition"
        );
    }
    match result {
        Ok(guard) => guard,
        Err(poisoned) => {
            tracing::warn!("mutex poisoned in {ctx}, recovering inner value");
            poisoned.into_inner()
        }
    }
}

/// Simple djb2 hash of a byte string — collision-resistant but not
/// cryptographic. Used to produce stable, unique file-name suffixes.
pub fn djb2(s: &str) -> u64 {
    let mut h: u64 = 5381;
    for b in s.bytes() {
        h = h.wrapping_mul(33).wrapping_add(b as u64);
    }
    h
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_djb2_empty_string_returns_seed() {
        assert_eq!(djb2(""), 5381);
    }

    #[test]
    fn test_djb2_same_input_same_output() {
        assert_eq!(djb2("hello"), djb2("hello"));
    }

    #[test]
    fn test_djb2_different_inputs_differ() {
        assert_ne!(djb2("foo"), djb2("bar"));
    }

    #[test]
    fn test_djb2_known_value() {
        // djb2("a") = 5381 * 33 + 97 = 177670
        assert_eq!(djb2("a"), 177670);
    }

    // ── Round 28: recover_lock, timed_lock ────────────────────────────────────

    #[test]
    fn test_recover_lock_returns_guard_for_healthy_mutex() {
        use std::sync::Mutex;
        let m = Mutex::new(42u32);
        let guard = recover_lock(m.lock(), "test");
        assert_eq!(*guard, 42);
    }

    #[test]
    fn test_recover_lock_recovers_from_poisoned_mutex() {
        use std::sync::{Arc, Mutex};
        let m = Arc::new(Mutex::new(99u32));
        let m2 = Arc::clone(&m);
        let _ = std::thread::spawn(move || {
            let _guard = m2.lock().unwrap();
            panic!("intentional panic to poison mutex");
        })
        .join();
        // Mutex is now poisoned; recover_lock should still return the guard
        let guard = recover_lock(m.lock(), "poisoned test");
        assert_eq!(*guard, 99);
    }

    #[test]
    fn test_timed_lock_returns_guard_for_healthy_mutex() {
        use std::sync::Mutex;
        let m = Mutex::new("hello");
        let guard = timed_lock(&m, "test");
        assert_eq!(*guard, "hello");
    }
}
