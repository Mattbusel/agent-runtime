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
