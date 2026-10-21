// Copyright (c) 2026 Ivan Tugay
// SPDX-License-Identifier: GPL-3.0-or-later
// Licensed under GPL-3.0 or later; see https://www.gnu.org/licenses/gpl-3.0.html

//! `runa tasks`: the claim registry (D18).

use std::path::PathBuf;

use runa_memory::{ClaimError, TaskRegistry};

use crate::cli::TaskAction;

fn task_registry_path() -> PathBuf {
    std::env::var("RUNA_TASK_REGISTRY")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from("docs/tasks.md"))
}

pub(super) fn cmd_tasks(action: TaskAction) -> Result<(), String> {
    let reg = TaskRegistry::open(task_registry_path());
    match action {
        TaskAction::List {} => {
            for id in reg.list_free().map_err(|e| e.to_string())? {
                println!("{id}");
            }
            Ok(())
        }
        TaskAction::Claim { task_id, agent } => match reg.claim(&task_id, &agent) {
            Ok(claim) => {
                println!(
                    "claimed {task_id} as {agent} at {started}",
                    started = claim.started_at
                );
                Ok(())
            }
            Err(ClaimError::AlreadyClaimed {
                agent: owner,
                started_at,
            }) => Err(format!(
                "{task_id}: in progress (owner: {owner}, since {started_at})"
            )),
            Err(e) => Err(e.to_string()),
        },
        TaskAction::Release { task_id, agent } => {
            reg.release(&task_id, &agent).map_err(|e| e.to_string())?;
            println!("released {task_id}");
            Ok(())
        }
    }
}
