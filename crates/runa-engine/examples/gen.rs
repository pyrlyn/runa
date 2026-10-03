// Copyright (c) 2026 Ivan Tugay
// SPDX-License-Identifier: GPL-3.0-or-later
// Licensed under GPL-3.0 or later; see https://www.gnu.org/licenses/gpl-3.0.html

//! P0.4 spike (+P0.6 baselines): load a GGUF via llama-cpp-2 and stream tokens.
//!
//! Usage: `cargo run -p runa-engine --example gen -- <model.gguf> [prompt] [n_predict]`
//!
//! Prints generated tokens to stdout and `pp` (prompt processing) + `tg`
//! (decode) speeds to stderr. On macOS the `metal` backend offloads all
//! layers; elsewhere it falls back to CPU.

use llama_cpp_2::context::params::LlamaContextParams;
use llama_cpp_2::llama_backend::LlamaBackend;
use llama_cpp_2::llama_batch::LlamaBatch;
use llama_cpp_2::model::AddBos;
use llama_cpp_2::model::LlamaModel;
use llama_cpp_2::model::params::LlamaModelParams;
use llama_cpp_2::sampling::LlamaSampler;
use std::io::Write;
use std::time::Instant;

fn main() {
    let mut args = std::env::args().skip(1);
    let model_path = args
        .next()
        .expect("usage: gen <model.gguf> [prompt] [n_predict]");
    let prompt = args.next().unwrap_or_else(|| "hi".to_string());
    let n_predict: i32 = args.next().and_then(|s| s.parse().ok()).unwrap_or(64);

    let backend = LlamaBackend::init().unwrap();
    let model_params = LlamaModelParams::default().with_n_gpu_layers(999);
    let model =
        LlamaModel::load_from_file(&backend, &model_path, &model_params).expect("load model");
    let ctx_params = LlamaContextParams::default();
    let mut ctx = model
        .new_context(&backend, ctx_params)
        .expect("create context");

    let prompt_tokens = model
        .str_to_token(&prompt, AddBos::Always)
        .expect("tokenize prompt");
    eprintln!("prompt: {} tokens", prompt_tokens.len());

    let mut batch = LlamaBatch::new(2048, 1);
    let last = prompt_tokens.len() as i32 - 1;
    for (i, tok) in (0_i32..).zip(prompt_tokens) {
        batch.add(tok, i, &[0], i == last).unwrap();
    }
    let t_pp = Instant::now();
    ctx.decode(&mut batch).expect("decode prompt");
    let pp_dt = t_pp.elapsed().as_secs_f64();
    eprintln!(
        "pp: {} tokens in {pp_dt:.2}s = {:.1} tok/s",
        last + 1,
        f64::from(last + 1) / pp_dt
    );

    let mut decoder = encoding_rs::UTF_8.new_decoder();
    let mut sampler = LlamaSampler::greedy();
    let t0 = Instant::now();
    let mut n_cur = batch.n_tokens();
    let mut n_gen = 0;
    while n_gen < n_predict {
        let tok = sampler.sample(&ctx, batch.n_tokens() - 1); // also accepts
        if tok == model.token_eos() {
            break;
        }
        let piece = model.token_to_piece(tok, &mut decoder, true, None).unwrap();
        print!("{piece}");
        std::io::stdout().flush().unwrap();
        batch.clear();
        batch.add(tok, n_cur, &[0], true).unwrap();
        ctx.decode(&mut batch).expect("decode token");
        n_cur += 1;
        n_gen += 1;
    }
    println!();
    let dt = t0.elapsed().as_secs_f64();
    eprintln!(
        "tg: {n_gen} tokens in {dt:.2}s = {:.1} tok/s",
        n_gen as f64 / dt
    );
}
