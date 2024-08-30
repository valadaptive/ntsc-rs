use std::{
    pin::Pin,
    sync::{Arc, Mutex, Weak},
    task::{Context, Poll, Wake, Waker},
};

use async_executor::{Executor, Task};
use eframe::egui;
use futures_lite::{Future, FutureExt};
use gstreamer::glib::clone::Downgrade;
use log::trace;

use super::AppFn;

struct AppExecutorInner {
    executor: Arc<Executor<'static>>,
    exec_future: Pin<Box<dyn Future<Output = ()> + Send>>,
    waker: Waker,
    egui_ctx: egui::Context,
    tasks: Vec<Task<Option<AppFn>>>,
}

impl AppExecutorInner {
    pub fn spawn(&mut self, future: impl Future<Output = Option<AppFn>> + 'static + Send) {
        trace!("spawned task on frame {}", self.egui_ctx.frame_nr());
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

        exec.tasks.retain_mut(|task| {
            if !task.is_finished() {
                return true;
            }
            trace!("finished task on frame {}", exec.egui_ctx.frame_nr());

            let Poll::Ready(cb) = task.poll(&mut context) else {
                panic!("task is finished but poll is not ready");
            };

            if let Some(cb) = cb {
                queued.push(cb);
            }

            false
        });

        queued
    }

    pub fn spawn(&self, future: impl Future<Output = Option<AppFn>> + 'static + Send) {
        let exec = &mut *self.0.lock().unwrap();
        exec.spawn(future);
    }

    pub fn make_spawner(&self) -> AppTaskSpawner {
        AppTaskSpawner(self.0.downgrade())
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
