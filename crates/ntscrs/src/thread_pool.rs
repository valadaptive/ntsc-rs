pub use rayon_core::ThreadPool;

pub fn with_thread_pool<T: Send>(op: impl (FnOnce() -> T) + Send + Sync) -> T {
    #[cfg(not(target_arch = "wasm32"))]
    {
        use std::sync::OnceLock;
        static POOL: OnceLock<ThreadPool> = OnceLock::new();

        let pool = POOL.get_or_init(|| {
            // On Windows debug builds, the stack overflows with the default stack size
            let mut pool = rayon_core::ThreadPoolBuilder::new().stack_size(2 * 1024 * 1024);

            // Use physical core count instead of logical core count. Hyperthreading seems to be ~20-25% slower, at least on
            // a Ryzen 7 7700X.
            if std::env::var("RAYON_NUM_THREADS").is_err() {
                pool = pool.num_threads(num_cpus::get_physical());
            }

            pool.build().unwrap()
        });

        pool.install(op)
    }
    #[cfg(target_arch = "wasm32")]
    {
        // wasm-bindgen-rayon doesn't support custom thread pools
        // https://github.com/RReverser/wasm-bindgen-rayon/issues/18
        op()
    }
}

pub struct ZipChunks<'a, const N: usize, T> {
    arrays: [&'a mut [T]; N],
    chunk_size: usize,
    start: usize,
}

impl<'a, const N: usize, T> ZipChunks<'a, N, T> {
    pub fn new(arrays: [&'a mut [T]; N], chunk_size: usize) -> Self {
        if let Some((first, rest)) = arrays.split_first() {
            for r in rest.iter() {
                assert_eq!(first.len(), r.len());
            }
        }

        Self {
            arrays,
            chunk_size,
            start: 0,
        }
    }

    pub fn len(&self) -> usize {
        match self.arrays.first() {
            Some(first) => first.len(),
            None => 0,
        }
    }

    fn split_at(mut self, chunk_idx: usize) -> (Self, Self) {
        let len = self.len();
        let split_point = (chunk_idx * self.chunk_size).min(len);

        // Store raw pointers before consuming self.arrays
        let ptrs: [*mut T; N] = self.arrays.each_mut().map(|s| s.as_mut_ptr());

        let left_halves = std::array::from_fn(|i: usize| unsafe {
            std::slice::from_raw_parts_mut(ptrs[i], split_point)
        });

        let right_halves = std::array::from_fn(|i: usize| unsafe {
            std::slice::from_raw_parts_mut(ptrs[i].add(split_point), len - split_point)
        });

        (
            Self {
                arrays: left_halves,
                chunk_size: self.chunk_size,
                start: self.start,
            },
            Self {
                arrays: right_halves,
                chunk_size: self.chunk_size,
                start: self.start + chunk_idx,
            },
        )
    }

    pub fn seq_for_each(mut self, mut cb: impl FnMut(usize, [&mut [T]; N])) {
        let len = self.len();
        let num_chunks = len / self.chunk_size;
        let remainder = len % self.chunk_size;
        let mut base = 0;
        for i in 0..num_chunks {
            let chunks: [&mut [T]; N] = self
                .arrays
                .each_mut()
                .map(|s| &mut s[base..base + self.chunk_size]);
            cb(i + self.start, chunks);

            base += self.chunk_size;
        }
        if remainder != 0 {
            let chunks: [&mut [T]; N] = self
                .arrays
                .each_mut()
                .map(|s| &mut s[base..base + remainder]);
            cb(self.start + num_chunks, chunks);
        }
    }

    fn par_for_each_inner<'b>(
        self,
        num_threads: usize,
        scope: &rayon_core::Scope<'b>,
        cb: &'b (impl Fn(usize, [&mut [T]; N]) + Send + Sync + Copy),
    ) where
        T: Send,
        'a: 'b,
    {
        let num_work_units = num_threads * 4;
        //const TARGET_GRANULARITY: usize = 2048;
        //let num_work_units = self.len().div_ceil(TARGET_GRANULARITY);

        // Number of chunks, including any partial chunk at the end
        let num_total_chunks = self.len().div_ceil(self.chunk_size);
        let chunks_per_thread = num_total_chunks / num_work_units;
        let mut remainder = num_total_chunks % num_work_units;
        if !self.len().is_multiple_of(self.chunk_size) {
            remainder += 1;
        }

        let n = if chunks_per_thread == 0 {
            remainder
        } else {
            num_work_units
        };

        let mut rest = self;
        for i in 0..n {
            let mut num_chunks = chunks_per_thread;
            if i < remainder {
                num_chunks += 1;
            }

            if num_chunks == 0 {
                break;
            }

            let (head, tail) = rest.split_at(num_chunks);
            rest = tail;

            scope.spawn(move |_| {
                head.seq_for_each(cb);
            });
        }
    }

    pub fn par_for_each(self, cb: impl Fn(usize, [&mut [T]; N]) + Send + Sync + Copy)
    where
        T: Send,
    {
        let num_threads = rayon_core::current_num_threads();
        if num_threads == 1 {
            self.seq_for_each(cb);
        } else {
            rayon_core::scope(|scope| {
                self.par_for_each_inner(num_threads, scope, &cb);
            });
        }
    }
}
