//! A tiny fork-join pool whose workers spin between jobs.
//!
//! One decode step runs ~150 short parallel regions (a matmul is 30-300 us).
//! rayon parks idle workers and wakes them with a futex, which cost ~50 us per
//! region here, more than half of each step. These workers spin for a moment
//! after each job, so back-to-back regions start in well under a microsecond,
//! and park only when the engine goes idle.

use std::cell::UnsafeCell;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering::*};
use std::sync::{Condvar, Mutex, OnceLock};
use std::time::{Duration, Instant};

const SPIN: Duration = Duration::from_millis(2);

type Job = *const (dyn Fn(usize) + Sync);

struct Shared {
    generation: AtomicU64,
    job: UnsafeCell<Option<Job>>,
    tasks: AtomicUsize,
    next: AtomicUsize,
    /// Workers allowed to join a job (the caller is one more).
    helpers: AtomicUsize,
    pending: AtomicUsize,
    sleepers: AtomicUsize,
    lock: Mutex<()>,
    wake: Condvar,
    /// One job at a time; a nested or concurrent call runs inline instead.
    busy: Mutex<()>,
}

// SAFETY: `job` is written only by the caller holding `busy`, before the
// Release store to `generation`, and read by workers only after an Acquire
// load observes that generation; the caller waits for `pending == 0` before
// clearing it, so no worker reads it afterwards.
unsafe impl Sync for Shared {}

struct Pool {
    shared: &'static Shared,
    workers: usize,
}

fn pool() -> &'static Pool {
    static POOL: OnceLock<Pool> = OnceLock::new();
    POOL.get_or_init(|| {
        let n = std::env::var("GLASSBOX_THREADS")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or_else(|| std::thread::available_parallelism().map_or(1, |n| n.get()))
            .max(1);
        let shared: &'static Shared = Box::leak(Box::new(Shared {
            generation: AtomicU64::new(0),
            job: UnsafeCell::new(None),
            tasks: AtomicUsize::new(0),
            next: AtomicUsize::new(0),
            helpers: AtomicUsize::new(n - 1),
            pending: AtomicUsize::new(0),
            sleepers: AtomicUsize::new(0),
            lock: Mutex::new(()),
            wake: Condvar::new(),
            busy: Mutex::new(()),
        }));
        for id in 0..n - 1 {
            std::thread::Builder::new()
                .name(format!("glassbox-{id}"))
                .spawn(move || worker(shared, id))
                .expect("spawn worker");
        }
        Pool { shared, workers: n - 1 }
    })
}

/// Threads used by `par_for`, the caller included.
pub fn threads() -> usize {
    pool().shared.helpers.load(Relaxed) + 1
}

/// Use at most `n` threads (at least 1, at most the pool size) from now on.
pub fn set_threads(n: usize) {
    let p = pool();
    p.shared.helpers.store(n.clamp(1, p.workers + 1) - 1, Relaxed);
}

fn run(s: &Shared, f: &(dyn Fn(usize) + Sync)) {
    let n = s.tasks.load(Relaxed);
    loop {
        let i = s.next.fetch_add(1, Relaxed);
        if i >= n {
            break;
        }
        f(i);
    }
}

fn worker(s: &'static Shared, id: usize) {
    let mut seen = 0;
    loop {
        let start = Instant::now();
        let mut spins = 0u32;
        loop {
            let g = s.generation.load(Acquire);
            if g != seen {
                seen = g;
                break;
            }
            spins += 1;
            if !spins.is_multiple_of(1024) || start.elapsed() < SPIN {
                std::hint::spin_loop();
                continue;
            }
            s.sleepers.fetch_add(1, SeqCst);
            let mut guard = s.lock.lock().unwrap();
            while s.generation.load(SeqCst) == seen {
                guard = s.wake.wait(guard).unwrap();
            }
            drop(guard);
            s.sleepers.fetch_sub(1, SeqCst);
        }
        if id >= s.helpers.load(Relaxed) {
            continue;
        }
        // SAFETY: see `unsafe impl Sync for Shared`: the job is set and stays
        // alive until this worker decrements `pending`.
        let job = unsafe { (*s.job.get()).expect("job posted") };
        // SAFETY: the pointer refers to the caller's closure, alive as above.
        run(s, unsafe { &*job });
        s.pending.fetch_sub(1, AcqRel);
    }
}

/// Run `f(0..n)` across the pool and return when every call has finished.
pub fn par_for(n: usize, f: impl Fn(usize) + Sync) {
    let p = pool();
    let s = p.shared;
    let helpers = s.helpers.load(Relaxed);
    let guard = if n > 1 && helpers > 0 {
        s.busy.try_lock().ok()
    } else {
        None
    };
    let Some(_guard) = guard else {
        (0..n).for_each(f);
        return;
    };
    let f: &(dyn Fn(usize) + Sync) = &f;
    // SAFETY: we erase the closure's lifetime to share it with the workers.
    // It outlives every use: we spin until `pending` reaches zero below,
    // and clear the slot before returning.
    let job: Job = unsafe { std::mem::transmute::<&(dyn Fn(usize) + Sync), &'static (dyn Fn(usize) + Sync)>(f) };
    // SAFETY: we hold `busy`, and no worker reads the slot until the
    // generation bump below publishes it.
    unsafe { *s.job.get() = Some(job) };
    s.tasks.store(n, Relaxed);
    s.next.store(0, Relaxed);
    s.pending.store(helpers, Relaxed);
    s.generation.fetch_add(1, SeqCst);
    if s.sleepers.load(SeqCst) > 0 {
        let _l = s.lock.lock().unwrap();
        s.wake.notify_all();
    }
    run(s, f);
    while s.pending.load(Acquire) != 0 {
        std::hint::spin_loop();
    }
    // SAFETY: every helper has finished with the job.
    unsafe { *s.job.get() = None };
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_index_runs_once() {
        for n in [0, 1, 2, 7, 100, 1000] {
            let hits: Vec<AtomicUsize> = (0..n).map(|_| AtomicUsize::new(0)).collect();
            par_for(n, |i| {
                hits[i].fetch_add(1, Relaxed);
            });
            assert!(hits.iter().all(|h| h.load(Relaxed) == 1));
        }
    }
}
