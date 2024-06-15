use std::{
    pin::Pin,
    sync::Arc,
    task::{Context, Poll, Waker},
};

use async_executor::Executor;
use async_task::Task;
use eframe::egui;
use futures_lite::{Future, FutureExt};
use log::trace;

use super::AppFn;

pub struct AppExecutor {
    executor: Arc<Executor<'static>>,
    exec_future: Pin<Box<dyn Future<Output = ()> + Send>>,
    waker: Waker,
    egui_ctx: egui::Context,
    tasks: Vec<Task<Option<AppFn>>>,
}

impl AppExecutor {
    pub fn new(egui_ctx: egui::Context) -> Self {
        let executor = Arc::new(Executor::new());
        let executor_for_future = Arc::clone(&executor);
        let egui_ctx_for_waker = egui_ctx.clone();
        Self {
            executor,
            exec_future: Box::pin(async move {
                executor_for_future
                    .run(futures_lite::future::pending())
                    .await
            }),
            waker: waker_fn::waker_fn(move || egui_ctx_for_waker.request_repaint()),
            egui_ctx,
            tasks: Vec::new(),
        }
    }

    #[must_use]
    pub fn tick(&mut self) -> Vec<AppFn> {
        let mut context = Context::from_waker(&self.waker);
        let _ = self.exec_future.poll(&mut context);

        let mut queued = Vec::new();

        self.tasks.retain_mut(|task| {
            if !task.is_finished() {
                return true;
            }
            trace!("finished task on frame {}", self.egui_ctx.frame_nr());

            let Poll::Ready(cb) = task.poll(&mut context) else {
                panic!("task is finished but poll is not ready");
            };

            if let Some(cb) = cb {
                queued.push(cb);
            }

            return false;
        });

        queued
    }

    pub fn spawn(&mut self, future: impl Future<Output = Option<AppFn>> + 'static + Send) {
        trace!("spawned task on frame {}", self.egui_ctx.frame_nr());
        let task = self.executor.spawn(future);
        self.tasks.push(task);
        self.egui_ctx.request_repaint();
    }
}
