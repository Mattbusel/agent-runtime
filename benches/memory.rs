use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use llm_agent_runtime::memory::{AgentId, EpisodicStore, WorkingMemory};

fn bench_episodic_store_add(c: &mut Criterion) {
    let store = EpisodicStore::new();
    let agent = AgentId::new("bench-agent");
    c.bench_function("EpisodicStore::add_episode", |b| {
        b.iter(|| {
            store
                .add_episode(black_box(agent.clone()), black_box("bench content"), black_box(0.5))
                .unwrap();
        });
    });
}

fn bench_episodic_store_recall(c: &mut Criterion) {
    let store = EpisodicStore::new();
    let agent = AgentId::new("bench-recall");
    for i in 0..100 {
        store
            .add_episode(agent.clone(), format!("memory item {i}"), 0.5)
            .unwrap();
    }

    let mut group = c.benchmark_group("EpisodicStore::recall");
    for limit in [10, 25, 50, 100] {
        group.bench_with_input(BenchmarkId::from_parameter(limit), &limit, |b, &limit| {
            b.iter(|| store.recall(black_box(&agent), black_box(limit)).unwrap());
        });
    }
    group.finish();
}

fn bench_working_memory_set_get(c: &mut Criterion) {
    let wm = WorkingMemory::new(100).unwrap();
    c.bench_function("WorkingMemory::set", |b| {
        let mut i = 0usize;
        b.iter(|| {
            wm.set(black_box(format!("key-{}", i % 50)), black_box("value"))
                .unwrap();
            i += 1;
        });
    });

    c.bench_function("WorkingMemory::get", |b| {
        wm.set("lookup-key", "some-value").unwrap();
        b.iter(|| wm.get(black_box("lookup-key")).unwrap());
    });
}

criterion_group!(benches, bench_episodic_store_add, bench_episodic_store_recall, bench_working_memory_set_get);
criterion_main!(benches);
