//! Demonstrates composing all four orchestrator primitives:
//! BackpressureGuard → Deduplicator → CircuitBreaker → RetryPolicy.
//!
//! Run with:
//!   cargo run --example orchestrator_composition --features orchestrator

#[cfg(feature = "orchestrator")]
fn main() {
    use llm_agent_runtime::orchestrator::{
        BackpressureGuard, CircuitBreaker, Deduplicator, DeduplicationResult, RetryPolicy,
    };
    use std::time::Duration;

    // 1. BackpressureGuard — shed requests when more than 4 are in-flight.
    let guard = BackpressureGuard::new(4).expect("valid capacity");

    // 2. Deduplicator — cache identical requests within a 5-second window.
    let dedup = Deduplicator::new(Duration::from_secs(5));

    // 3. CircuitBreaker — open after 3 consecutive failures; probe after 10 s.
    let cb = CircuitBreaker::new("llm-api", 3, Duration::from_secs(10))
        .expect("valid threshold");

    // 4. RetryPolicy — up to 3 attempts with 50 ms base delay.
    let retry = RetryPolicy::exponential(3, 50).expect("valid retry policy");

    // Simulate sending a request through the full stack.
    let request_key = "summarise:doc-42";
    let request_body = "Summarise this document…";

    // Layer 1: backpressure
    if let Err(e) = guard.try_acquire() {
        eprintln!("Request shed — too many in-flight: {e}");
        return;
    }

    // Layer 2: deduplication
    match dedup.check_and_register(request_key).expect("dedup check") {
        DeduplicationResult::Cached(result) => {
            println!("Dedup cache hit: {result}");
            guard.release().ok();
            return;
        }
        DeduplicationResult::InProgress => {
            println!("Duplicate request already in-flight, dropping");
            guard.release().ok();
            return;
        }
        DeduplicationResult::New => {}
    }

    // Layer 3 + 4: circuit breaker wrapping retry
    let mut attempt = 0u32;
    let result = loop {
        attempt += 1;
        let outcome = cb.call(|| {
            // Simulate a flaky LLM API call.
            if attempt < 3 {
                Err(format!("API error on attempt {attempt}"))
            } else {
                Ok(format!("Summary of: {request_body}"))
            }
        });

        match outcome {
            Ok(val) => break Ok(val),
            Err(e) if attempt < retry.max_attempts => {
                let delay = retry.delay_for(attempt);
                println!("Attempt {attempt} failed ({e}), retrying in {delay:?}");
                std::thread::sleep(delay);
            }
            Err(e) => break Err(e),
        }
    };

    // Complete deduplication entry.
    match &result {
        Ok(val) => {
            dedup.complete(request_key, val.clone()).ok();
        }
        Err(_) => {
            // No evict method; the in-flight entry will expire via TTL.
        }
    }

    // Release backpressure slot.
    guard.release().ok();

    match result {
        Ok(summary) => println!("Success: {summary}"),
        Err(e) => eprintln!("All attempts failed: {e}"),
    }
}

#[cfg(not(feature = "orchestrator"))]
fn main() {
    eprintln!("This example requires --features orchestrator");
}
