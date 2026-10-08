// Copyright (c) 2026 Ivan Tugay
// SPDX-License-Identifier: GPL-3.0-or-later
// Licensed under GPL-3.0 or later; see https://www.gnu.org/licenses/gpl-3.0.html

//! P16.2: a program other than the `runa` binary drives the model pool
//! through the library only. The engine is a scripted thread behind the
//! real [`EngineJob`] protocol, so no weights are needed; the ggml and
//! mistral engines are covered by `runa`'s own serve/daemon e2e tests.

use std::path::PathBuf;
use std::sync::mpsc::{Receiver, Sender, channel};
use std::sync::{Arc, Mutex};

use runa_pool::{
    BackendKind, EngineJob, GenEvent, GenerateRequest, LoadConfig, ModelPool, Placement, Placer,
    StopReason, Streamed, fixed_placer, generate, generate_stream,
};

/// An engine thread that answers every request with "hello".
fn scripted_engine() -> Sender<EngineJob> {
    let (jobs, rx): (_, Receiver<EngineJob>) = channel();
    std::thread::spawn(move || {
        while let Ok(job) = rx.recv() {
            match job {
                EngineJob::Generate { resp, .. } => {
                    let _ = resp.send(Ok(vec![
                        GenEvent::Text("hel".into()),
                        GenEvent::Text("lo".into()),
                        GenEvent::Done(StopReason::Eos),
                    ]));
                }
                EngineJob::GenerateStream { tx, .. } => {
                    let _ = tx.blocking_send(Ok(Streamed::Prompt(3)));
                    let _ = tx.blocking_send(Ok(Streamed::Event(GenEvent::Text("hel".into()))));
                    let _ = tx.blocking_send(Ok(Streamed::Event(GenEvent::Text("lo".into()))));
                    let _ = tx.blocking_send(Ok(Streamed::Event(GenEvent::Done(StopReason::Eos))));
                }
                EngineJob::Embed { .. } | EngineJob::Idle => {}
            }
        }
    });
    jobs
}

/// The pool only checks that the path exists and ends in `.gguf`.
fn model_file() -> tempfile::NamedTempFile {
    tempfile::Builder::new().suffix(".gguf").tempfile().unwrap()
}

fn pool_with(model: &tempfile::NamedTempFile, placer: Placer) -> Arc<Mutex<ModelPool>> {
    let path: PathBuf = model.path().to_owned();
    let pool = ModelPool::new(
        vec![("fake".into(), path)],
        BackendKind::Gguf,
        LoadConfig::default(),
        1,
        placer,
    )
    .expect("pool over an existing file");
    Arc::new(Mutex::new(pool))
}

#[tokio::test]
async fn generates_events_through_the_library() {
    let model = model_file();
    let pool = pool_with(&model, fixed_placer(Placement::cpu()));
    pool.lock()
        .unwrap()
        .attach_engine("fake", scripted_engine())
        .unwrap();

    let events = generate(&pool, "fake", GenerateRequest::default())
        .await
        .expect("generation");
    assert_eq!(
        events,
        [
            GenEvent::Text("hel".into()),
            GenEvent::Text("lo".into()),
            GenEvent::Done(StopReason::Eos),
        ]
    );
}

#[tokio::test]
async fn streams_events_through_the_library() {
    let model = model_file();
    let pool = pool_with(&model, fixed_placer(Placement::cpu()));
    pool.lock()
        .unwrap()
        .attach_engine("fake", scripted_engine())
        .unwrap();

    let mut rx = generate_stream(&pool, "fake", GenerateRequest::default(), 4)
        .await
        .expect("stream");
    let mut text = String::new();
    let mut prompt = None;
    while let Some(item) = rx.recv().await {
        match item.expect("no engine error") {
            Streamed::Prompt(n) => prompt = Some(n),
            Streamed::Event(GenEvent::Text(t)) => text.push_str(&t),
            Streamed::Event(_) => {}
        }
    }
    assert_eq!((prompt, text.as_str()), (Some(3), "hello"));
}

#[tokio::test]
async fn the_placer_decides_whether_a_model_loads() {
    let model = model_file();
    let refuse: Placer = Arc::new(|_, _| Err("unfit: no room".into()));
    let pool = pool_with(&model, refuse);

    let err = generate(&pool, "fake", GenerateRequest::default())
        .await
        .expect_err("placer refuses");
    assert_eq!(err, "unfit: no room");
    assert!(
        pool.lock()
            .unwrap()
            .attach_engine("missing", scripted_engine())
            .is_err(),
        "attach needs a model id the pool knows"
    );
}
