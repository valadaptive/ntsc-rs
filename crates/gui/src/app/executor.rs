use std::{
    pin::Pin,
    sync::{Arc, Mutex, Weak},
    task::{Context, Poll, Wake, Waker},
};

use async_executor::{Executor, Task};
use eframe::egui;
use futures_lite::{Future, FutureExt};
use log::trace;

use super::{AppFn, ApplessFn, NtscApp};

struct AppExecutorInner {
    executor: Arc<Executor<'static>>,
    exec_future: Pin<Box<dyn Future<Output = ()> + Send>>,
    waker: Waker,
    egui_ctx: egui::Context,
    tasks: Vec<Task<Option<AppFn>>>,
}

impl AppExecutorInner {
    pub fn spawn(&mut self, future: impl Future<Output = Option<AppFn>> + 'static + Send) {
        trace!(
            "spawned task on frame {}",
            self.egui_ctx.cumulative_pass_nr()
        );
        let task = self.executor.spawn(future);
        self.tasks.push(task);
        self.egui_ctx.request_repaint();
    }
}

struct AppWaker(egui::Context);

impl Wake for AppWaker {
    fn wake(self: Arc<Self>) {
        self.0.request_repaint();
    }

    fn wake_by_ref(self: &Arc<Self>) {
        self.0.request_repaint();
    }
}

pub struct AppExecutor(Arc<Mutex<AppExecutorInner>>);

impl AppExecutor {
    pub fn new(egui_ctx: egui::Context) -> Self {
        let executor = Arc::new(Executor::new());
        let executor_for_future = Arc::clone(&executor);
        let egui_ctx_for_waker = egui_ctx.clone();
        Self(Arc::new(Mutex::new(AppExecutorInner {
            executor,
            exec_future: Box::pin(async move {
                executor_for_future
                    .run(futures_lite::future::pending())
                    .await
            }),
            waker: Waker::from(Arc::new(AppWaker(egui_ctx_for_waker))),
            egui_ctx,
            tasks: Vec::new(),
        })))
    }

    #[must_use]
    pub fn tick(&self) -> Vec<AppFn> {
        let exec = &mut *self.0.lock().unwrap();

        let mut context = Context::from_waker(&exec.waker);
        let _ = exec.exec_future.poll(&mut context);

        let mut queued = Vec::new();

        exec.tasks.retain_mut(|task| match task.poll(&mut context) {
            Poll::Ready(cb) => {
                trace!(
                    "finished task on frame {}",
                    exec.egui_ctx.cumulative_pass_nr()
                );
                if let Some(cb) = cb {
                    queued.push(cb);
                }
                false
            }
            Poll::Pending => true,
        });

        queued
    }

    pub fn spawn(&self, future: impl Future<Output = Option<AppFn>> + 'static + Send) {
        let exec = &mut *self.0.lock().unwrap();
        exec.spawn(future);
    }

    pub fn make_spawner(&self) -> AppTaskSpawner {
        AppTaskSpawner(Arc::downgrade(&self.0))
    }
}

#[derive(Clone)]
pub struct AppTaskSpawner(Weak<Mutex<AppExecutorInner>>);

impl AppTaskSpawner {
    pub fn spawn(&self, future: impl Future<Output = Option<AppFn>> + 'static + Send) {
        let Some(exec) = self.0.upgrade() else {
            return;
        };

        exec.lock().unwrap().spawn(future);
    }
}

pub trait ApplessExecutor: Send + Sync {
    fn spawn(&self, future: impl Future<Output = Option<ApplessFn>> + 'static + Send);
}

impl ApplessExecutor for AppExecutor {
    fn spawn(&self, future: impl Future<Output = Option<ApplessFn>> + 'static + Send) {
        self.spawn(async { future.await.map(|cb| Box::new(|_: &mut NtscApp| cb()) as _) });
    }
}

impl ApplessExecutor for AppTaskSpawner {
    fn spawn(&self, future: impl Future<Output = Option<ApplessFn>> + 'static + Send) {
        self.spawn(async { future.await.map(|cb| Box::new(|_: &mut NtscApp| cb()) as _) });
    }
}
