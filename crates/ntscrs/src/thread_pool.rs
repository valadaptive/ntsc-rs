pub fn with_thread_pool<T: Send>(op: impl (FnOnce() -> T) + Send + Sync) -> T {
    #[cfg(not(target_arch = "wasm32"))]
    {
        use std::sync::OnceLock;
        static POOL: OnceLock<rayon::ThreadPool> = OnceLock::new();

        let pool = POOL.get_or_init(|| {
            // On Windows debug builds, the stack overflows with the default stack size
            let mut pool = rayon::ThreadPoolBuilder::new().stack_size(2 * 1024 * 1024);

            // Use physical core count instead of logical core count. Hyperthreading seems to be ~20-25% slower, at least on
            // a Ryzen 7 7700X.
            if std::env::var("RAYON_NUM_THREADS").is_err() {
                pool = pool.num_threads(num_cpus::get_physical());
            }

            pool.build().unwrap()
        });

        pool.scope(|_| op())
    }
    #[cfg(target_arch = "wasm32")]
    {
        // wasm-bindgen-rayon doesn't support custom thread pools
        // https://github.com/RReverser/wasm-bindgen-rayon/issues/18
        op()
    }
}
