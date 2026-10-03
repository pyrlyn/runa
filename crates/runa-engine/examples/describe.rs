// Copyright (c) 2026 Ivan Tugay
// SPDX-License-Identifier: GPL-3.0-or-later
// Licensed under GPL-3.0 or later; see https://www.gnu.org/licenses/gpl-3.0.html

//! P0.5 spike: load an mmproj via llama-cpp-2 `mtmd` and answer
//! "what is in this image?".
//!
//! Usage: `cargo run -p runa-engine --features mtmd --example describe -- \
//!   <model.gguf> <mmproj.gguf> <image> [prompt]`
//!
//! Requires the `mtmd` feature (skipped on targets without it).

use llama_cpp_2::context::params::LlamaContextParams;
use llama_cpp_2::llama_backend::LlamaBackend;
use llama_cpp_2::llama_batch::LlamaBatch;
use llama_cpp_2::model::params::LlamaModelParams;
use llama_cpp_2::model::{LlamaChatMessage, LlamaModel};
use llama_cpp_2::mtmd::{MtmdBitmap, MtmdContext, MtmdContextParams, MtmdInputText};
use llama_cpp_2::sampling::LlamaSampler;
use std::ffi::CString;
use std::io::Write;
use std::time::Instant;

const N_GEN: i32 = 64;

fn main() {
    let mut args = std::env::args().skip(1);
    let model_path = args
        .next()
        .expect("usage: describe <model> <mmproj> <image> [prompt]");
    let mmproj_path = args
        .next()
        .expect("usage: describe <model> <mmproj> <image> [prompt]");
    let image_path = args
        .next()
        .expect("usage: describe <model> <mmproj> <image> [prompt]");
    let prompt = args
        .next()
        .unwrap_or_else(|| "What is in this image?".to_string());

    let backend = LlamaBackend::init().unwrap();
    let model_params = LlamaModelParams::default().with_n_gpu_layers(999);
    let model =
        LlamaModel::load_from_file(&backend, &model_path, &model_params).expect("load model");
    let mut ctx = model
        .new_context(&backend, LlamaContextParams::default())
        .expect("create context");

    let mtmd_params = MtmdContextParams {
        use_gpu: true,
        print_timings: false,
        n_threads: 4,
        media_marker: CString::new(llama_cpp_2::mtmd::mtmd_default_marker()).unwrap(),
        image_min_tokens: -1,
        image_max_tokens: -1,
    };
    let mtmd_ctx =
        MtmdContext::init_from_file(&mmproj_path, &model, &mtmd_params).expect("init mmproj");
    assert!(mtmd_ctx.support_vision(), "mmproj has no vision support");
    eprintln!("mmproj loaded; vision supported");

    let bitmap = MtmdBitmap::from_file(&mtmd_ctx, &image_path, false).expect("load image");

    // Prefer the model's chat template; fall back to raw text with marker.
    let marker = llama_cpp_2::mtmd::mtmd_default_marker();
    let text = match model.chat_template(None) {
        Ok(tpl) => {
            let chat = vec![
                LlamaChatMessage::new("user".to_string(), format!("{prompt} {marker}"))
                    .expect("chat message"),
            ];
            model
                .apply_chat_template(&tpl, &chat, true)
                .expect("apply chat template")
        }
        Err(_) => format!("{prompt} {marker}"),
    };
    let chunks = mtmd_ctx
        .tokenize(
            MtmdInputText {
                text,
                add_special: true,
                parse_special: true,
            },
            &[&bitmap],
        )
        .expect("mtmd tokenize");
    eprintln!("tokenized into {} chunks", chunks.len());

    let mut n_past = chunks
        .eval_chunks(&mtmd_ctx, &ctx, 0, 0, 512, true)
        .expect("eval chunks");

    let mut batch = LlamaBatch::new(512, 1);
    let mut sampler = LlamaSampler::greedy();
    let mut decoder = encoding_rs::UTF_8.new_decoder();
    let t0 = Instant::now();
    let mut n_gen = 0;
    while n_gen < N_GEN {
        let tok = sampler.sample(&ctx, -1); // also accepts
        if model.token_eos() == tok {
            break;
        }
        let piece = model.token_to_piece(tok, &mut decoder, true, None).unwrap();
        print!("{piece}");
        std::io::stdout().flush().unwrap();
        batch.clear();
        batch.add(tok, n_past, &[0], true).unwrap();
        ctx.decode(&mut batch).expect("decode token");
        n_past += 1;
        n_gen += 1;
    }
    println!();
    let dt = t0.elapsed().as_secs_f64();
    eprintln!(
        "gen: {n_gen} tokens in {dt:.2}s = {:.1} tok/s",
        n_gen as f64 / dt
    );
}
