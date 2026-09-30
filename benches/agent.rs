use criterion::{black_box, criterion_group, criterion_main, Criterion};
use llm_agent_runtime::agent::{parse_react_step, ToolRegistry, ToolSpec};

fn bench_parse_react_step(c: &mut Criterion) {
    let input = "Thought: I should look up the answer\nAction: search {\"q\": \"rust benchmarks\"}";
    c.bench_function("parse_react_step", |b| {
        b.iter(|| parse_react_step(black_box(input)).unwrap());
    });
}

fn bench_tool_registry_call_hit(c: &mut Criterion) {
    let rt = tokio::runtime::Runtime::new().unwrap();
    let mut registry = ToolRegistry::new();
    registry.register(ToolSpec::new("echo", "Echo args", |args| args.clone()));

    c.bench_function("ToolRegistry::call (hit)", |b| {
        b.iter(|| {
            rt.block_on(async {
                registry
                    .call(black_box("echo"), black_box(serde_json::json!({"x": 1})))
                    .await
                    .unwrap()
            })
        });
    });
}

fn bench_tool_registry_call_miss(c: &mut Criterion) {
    let rt = tokio::runtime::Runtime::new().unwrap();
    let mut registry = ToolRegistry::new();
    for i in 0..20 {
        registry.register(ToolSpec::new(
            format!("tool_{i}"),
            "A tool",
            |args| args.clone(),
        ));
    }

    c.bench_function("ToolRegistry::call (miss, 20 tools)", |b| {
        b.iter(|| {
            rt.block_on(async {
                let _ = registry
                    .call(black_box("nonexistent"), black_box(serde_json::json!({})))
                    .await;
            })
        });
    });
}

criterion_group!(benches, bench_parse_react_step, bench_tool_registry_call_hit, bench_tool_registry_call_miss);
criterion_main!(benches);
