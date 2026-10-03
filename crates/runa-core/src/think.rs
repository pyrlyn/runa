// Copyright (c) 2026 Ivan Tugay
// SPDX-License-Identifier: GPL-3.0-or-later
// Licensed under GPL-3.0 or later; see https://www.gnu.org/licenses/gpl-3.0.html

//! Thinking primitive (plan D7, P3.1) plus local budget forcing (P3.3).

use std::fmt;

/// Default grace tokens when a budget is set without an explicit grace.
pub const DEFAULT_GRACE: u32 = 64;

/// Cloud/local effort knob. Local mapping to token budgets is P3.4.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Effort {
    Low,
    Medium,
    High,
    Max,
}

impl Effort {
    /// Parse `low|medium|high|max` (case-insensitive).
    pub fn parse(s: &str) -> Result<Self, String> {
        match s.trim().to_ascii_lowercase().as_str() {
            "low" => Ok(Effort::Low),
            "medium" => Ok(Effort::Medium),
            "high" => Ok(Effort::High),
            "max" => Ok(Effort::Max),
            other => Err(format!(
                "{other}: --effort must be low | medium | high | max"
            )),
        }
    }
}

impl fmt::Display for Effort {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Effort::Low => "low",
            Effort::Medium => "medium",
            Effort::High => "high",
            Effort::Max => "max",
        })
    }
}

/// Default reasoning caps per effort (plan P3.4).
pub const EFFORT_BUDGET_LOW: u32 = 512;
pub const EFFORT_BUDGET_MEDIUM: u32 = 2048;
pub const EFFORT_BUDGET_HIGH: u32 = 8192;

impl Effort {
    /// Map effort → reasoning token budget given `remaining_ctx`.
    /// `Max` → `None` (unlimited).
    pub fn budget_tokens(self, remaining_ctx: u32) -> Option<u32> {
        let cap = match self {
            Effort::Low => EFFORT_BUDGET_LOW,
            Effort::Medium => EFFORT_BUDGET_MEDIUM,
            Effort::High => EFFORT_BUDGET_HIGH,
            Effort::Max => return None,
        };
        Some(cap.min(remaining_ctx))
    }
}

/// Model-specific system hints for thinking (plan P3.4).
pub fn effort_system_hint(model_id: &str, effort: Effort) -> Option<String> {
    let id = model_id.to_ascii_lowercase();
    if id.contains("gpt-oss") || id.contains("gpt_oss") {
        let level = match effort {
            Effort::Low => "low",
            Effort::Medium => "medium",
            Effort::High | Effort::Max => "high",
        };
        return Some(format!("Reasoning: {level}"));
    }
    None
}

/// How thinking is enabled for a request.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ThinkMode {
    Off,
    On,
    Budget { tokens: u32, grace: u32 },
    Effort(Effort),
}

/// Request-level thinking config (local and cloud share this).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ThinkConfig {
    pub mode: ThinkMode,
    /// Surface reasoning to the user (`--show-reasoning`).
    pub show: bool,
}

impl Default for ThinkConfig {
    fn default() -> Self {
        ThinkConfig {
            mode: ThinkMode::Off,
            show: false,
        }
    }
}

impl fmt::Display for ThinkConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.mode {
            ThinkMode::Off => write!(f, "off")?,
            ThinkMode::On => write!(f, "on")?,
            ThinkMode::Budget { tokens, grace } => write!(f, "budget {tokens} grace {grace}")?,
            ThinkMode::Effort(e) => write!(f, "effort {e}")?,
        }
        if self.show {
            write!(f, " show")?;
        }
        Ok(())
    }
}

/// Sparse overlay from CLI flags, env, `[think]`, or `/think`.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ThinkOverrides {
    /// `Some(true)` = on, `Some(false)` = off.
    pub think: Option<bool>,
    pub budget: Option<u32>,
    pub grace: Option<u32>,
    pub effort: Option<Effort>,
    pub show: Option<bool>,
}

impl ThinkOverrides {
    /// Parse `--think on|off`.
    pub fn parse_think(s: &str) -> Result<bool, String> {
        match s.trim().to_ascii_lowercase().as_str() {
            "on" | "true" | "1" => Ok(true),
            "off" | "false" | "0" => Ok(false),
            other => Err(format!("{other}: --think must be on | off")),
        }
    }

    /// Parse `--show-reasoning` / `RUNA_SHOW_REASONING`.
    pub fn parse_show(s: &str) -> Result<bool, String> {
        match s.trim().to_ascii_lowercase().as_str() {
            "on" | "true" | "1" | "yes" => Ok(true),
            "off" | "false" | "0" | "no" => Ok(false),
            other => Err(format!("{other}: show-reasoning must be true | false")),
        }
    }

    /// `/think on`, `/think budget 2048 show`, `/think effort high`.
    pub fn from_slash_args(args: &str) -> Result<Self, String> {
        let mut o = ThinkOverrides::default();
        let mut it = args.split_whitespace().peekable();
        if args.trim().is_empty() {
            return Ok(o);
        }
        while let Some(tok) = it.next() {
            match tok {
                "on" => o.think = Some(true),
                "off" => o.think = Some(false),
                "show" => o.show = Some(true),
                "hide" => o.show = Some(false),
                "budget" => {
                    let n = it.next().ok_or("usage: /think budget <n>")?;
                    o.budget = Some(parse_budget(n)?);
                }
                "effort" => {
                    let e = it
                        .next()
                        .ok_or("usage: /think effort <low|medium|high|max>")?;
                    o.effort = Some(Effort::parse(e)?);
                }
                "grace" => {
                    let n = it.next().ok_or("usage: /think grace <n>")?;
                    o.grace = Some(parse_u32(n, "grace")?);
                }
                other => {
                    return Err(format!(
                        "unknown /think arg {other:?} (on|off|budget|effort|grace|show|hide)"
                    ));
                }
            }
        }
        Ok(o)
    }
}

impl ThinkConfig {
    /// Effective local reasoning budget (P3.4). `None` = unlimited / model default.
    pub fn reasoning_budget(self, remaining_ctx: u32) -> Option<u32> {
        match self.mode {
            ThinkMode::Off => None,
            ThinkMode::On => None,
            ThinkMode::Budget { tokens, .. } => Some(tokens.min(remaining_ctx)),
            ThinkMode::Effort(e) => e.budget_tokens(remaining_ctx),
        }
    }

    /// Overlay flags/config onto this base. Budget and effort override
    /// `--think on`; `--think off` cannot mix with either.
    pub fn apply(&self, o: &ThinkOverrides) -> Result<ThinkConfig, String> {
        let show = o.show.unwrap_or(self.show);
        if o.think == Some(false) && (o.budget.is_some() || o.effort.is_some()) {
            return Err("--think off cannot be combined with --think-budget or --effort".into());
        }
        if o.budget.is_some() && o.effort.is_some() {
            return Err("--think-budget and --effort cannot be combined".into());
        }
        let grace_base = match self.mode {
            ThinkMode::Budget { grace, .. } => grace,
            _ => DEFAULT_GRACE,
        };
        let grace = o.grace.unwrap_or(grace_base);
        let mode = if let Some(tokens) = o.budget {
            ThinkMode::Budget { tokens, grace }
        } else if let Some(e) = o.effort {
            ThinkMode::Effort(e)
        } else if let Some(on) = o.think {
            if on { ThinkMode::On } else { ThinkMode::Off }
        } else {
            match self.mode {
                ThinkMode::Budget { tokens, grace: g } => ThinkMode::Budget {
                    tokens,
                    grace: o.grace.unwrap_or(g),
                },
                other => other,
            }
        };
        Ok(ThinkConfig { mode, show })
    }
}

/// What the sampler should do on the next decode step (P3.3).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ForceKind {
    /// Unmodified sampling.
    Sample,
    /// Sample, but boost the close-tag first token.
    Bias,
    /// Do not sample; emit the next injected token.
    Force,
}

/// Message prepended to the close tag when the hard budget is hit.
pub const BUDGET_MESSAGE: &str = "Answer now.";

/// Logit added to the close token during the grace window.
pub const CLOSE_LOGIT_BIAS: f32 = 10.0;

/// Counts reasoning tokens after think-open and decides bias vs inject.
#[derive(Debug, Clone)]
pub struct BudgetClock {
    budget: u32,
    grace: u32,
    counted: u32,
    was_in_reason: bool,
    forcing: bool,
    done: bool,
}

impl BudgetClock {
    pub fn new(budget: u32, grace: u32) -> Self {
        BudgetClock {
            budget,
            grace,
            counted: 0,
            was_in_reason: false,
            forcing: false,
            done: false,
        }
    }

    /// `None` when thinking is unlimited (`Off` / `On` / `Effort::Max`).
    pub fn from_think(think: ThinkConfig, remaining_ctx: u32) -> Option<Self> {
        let budget = think.reasoning_budget(remaining_ctx)?;
        let grace = match think.mode {
            ThinkMode::Budget { grace, .. } => grace,
            _ => DEFAULT_GRACE,
        };
        Some(BudgetClock::new(budget, grace))
    }

    /// Reasoning-body tokens observed so far (open tag itself is not counted).
    pub fn counted(&self) -> u32 {
        self.counted
    }

    /// Token count at which close-tag logit bias starts (`budget − grace`).
    pub fn bias_at(&self) -> u32 {
        self.budget.saturating_sub(self.grace)
    }

    /// After a generated token has been parsed.
    pub fn observe(&mut self, in_reason: bool, holding_partial: bool) {
        if self.done {
            // Re-arm on a new think-open (models may emit several blocks).
            if in_reason && !self.was_in_reason {
                let budget = self.budget;
                let grace = self.grace;
                *self = BudgetClock::new(budget, grace);
                self.was_in_reason = true;
            } else {
                self.was_in_reason = in_reason;
            }
            return;
        }
        if self.forcing {
            if !in_reason {
                self.done = true;
                self.forcing = false;
            }
            self.was_in_reason = in_reason;
            return;
        }
        if in_reason && self.was_in_reason {
            self.counted = self.counted.saturating_add(1);
            if self.counted >= self.budget && !holding_partial {
                self.forcing = true;
            }
        } else if self.was_in_reason && !in_reason {
            self.done = true;
        }
        self.was_in_reason = in_reason;
    }

    pub fn kind(&self, in_reason: bool, holding_partial: bool) -> ForceKind {
        if self.done || !in_reason {
            return ForceKind::Sample;
        }
        if (self.forcing || self.counted >= self.budget) && !holding_partial {
            return ForceKind::Force;
        }
        if self.counted >= self.bias_at() {
            return ForceKind::Bias;
        }
        ForceKind::Sample
    }
}

pub fn parse_budget(s: &str) -> Result<u32, String> {
    let n = parse_u32(s, "think-budget")?;
    if n == 0 {
        return Err("--think-budget must be > 0".into());
    }
    Ok(n)
}

fn parse_u32(s: &str, what: &str) -> Result<u32, String> {
    s.parse::<u32>()
        .map_err(|_| format!("{s}: {what} must be a positive integer"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn effort_budget_table() {
        assert_eq!(Effort::Low.budget_tokens(10_000), Some(512));
        assert_eq!(Effort::Medium.budget_tokens(10_000), Some(2048));
        assert_eq!(Effort::High.budget_tokens(10_000), Some(8192));
        assert_eq!(Effort::Max.budget_tokens(10_000), None);
        assert_eq!(Effort::Low.budget_tokens(256), Some(256));
    }

    #[test]
    fn effort_system_hint_gpt_oss() {
        assert_eq!(
            effort_system_hint("gpt-oss-20b", Effort::High).as_deref(),
            Some("Reasoning: high")
        );
        assert_eq!(effort_system_hint("Qwen3-8B", Effort::High), None);
    }

    #[test]
    fn reasoning_budget_from_config() {
        let cfg = ThinkConfig {
            mode: ThinkMode::Effort(Effort::Medium),
            show: false,
        };
        assert_eq!(cfg.reasoning_budget(4096), Some(2048));
        let cfg = ThinkConfig {
            mode: ThinkMode::Budget {
                tokens: 3000,
                grace: DEFAULT_GRACE,
            },
            show: false,
        };
        assert_eq!(cfg.reasoning_budget(2048), Some(2048));
    }

    #[test]
    fn effort_parse() {
        assert_eq!(Effort::parse("LOW").unwrap(), Effort::Low);
        assert_eq!(Effort::parse("medium").unwrap(), Effort::Medium);
        assert_eq!(Effort::parse("High").unwrap(), Effort::High);
        assert_eq!(Effort::parse("max").unwrap(), Effort::Max);
        assert!(Effort::parse("extreme").is_err());
    }

    #[test]
    fn think_on_off() {
        assert!(ThinkOverrides::parse_think("on").unwrap());
        assert!(!ThinkOverrides::parse_think("OFF").unwrap());
        assert!(ThinkOverrides::parse_think("maybe").is_err());
    }

    #[test]
    fn apply_budget_overrides_on() {
        let cfg = ThinkConfig {
            mode: ThinkMode::On,
            show: false,
        };
        let got = cfg
            .apply(&ThinkOverrides {
                think: Some(true),
                budget: Some(2048),
                grace: Some(32),
                ..ThinkOverrides::default()
            })
            .unwrap();
        assert_eq!(
            got.mode,
            ThinkMode::Budget {
                tokens: 2048,
                grace: 32
            }
        );
    }

    #[test]
    fn apply_effort() {
        let got = ThinkConfig::default()
            .apply(&ThinkOverrides {
                effort: Some(Effort::High),
                show: Some(true),
                ..ThinkOverrides::default()
            })
            .unwrap();
        assert_eq!(got.mode, ThinkMode::Effort(Effort::High));
        assert!(got.show);
    }

    #[test]
    fn apply_off_with_budget_errors() {
        let err = ThinkConfig::default()
            .apply(&ThinkOverrides {
                think: Some(false),
                budget: Some(256),
                ..ThinkOverrides::default()
            })
            .unwrap_err();
        assert!(err.contains("off"));
    }

    #[test]
    fn apply_budget_and_effort_errors() {
        assert!(
            ThinkConfig::default()
                .apply(&ThinkOverrides {
                    budget: Some(256),
                    effort: Some(Effort::Low),
                    ..ThinkOverrides::default()
                })
                .is_err()
        );
    }

    #[test]
    fn apply_zero_budget_rejected_at_parse() {
        assert!(parse_budget("0").is_err());
        assert_eq!(parse_budget("2048").unwrap(), 2048);
    }

    #[test]
    fn default_grace_when_budget_omits_it() {
        let got = ThinkConfig::default()
            .apply(&ThinkOverrides {
                budget: Some(512),
                ..ThinkOverrides::default()
            })
            .unwrap();
        assert_eq!(
            got.mode,
            ThinkMode::Budget {
                tokens: 512,
                grace: DEFAULT_GRACE
            }
        );
    }

    #[test]
    fn slash_args() {
        let o = ThinkOverrides::from_slash_args("budget 2048 show").unwrap();
        assert_eq!(o.budget, Some(2048));
        assert_eq!(o.show, Some(true));
        let o = ThinkOverrides::from_slash_args("effort max hide").unwrap();
        assert_eq!(o.effort, Some(Effort::Max));
        assert_eq!(o.show, Some(false));
        assert!(ThinkOverrides::from_slash_args("nope").is_err());
        assert!(ThinkOverrides::from_slash_args("budget").is_err());
    }

    #[test]
    fn display_roundtrip_shape() {
        let c = ThinkConfig {
            mode: ThinkMode::Budget {
                tokens: 2048,
                grace: 64,
            },
            show: true,
        };
        assert_eq!(c.to_string(), "budget 2048 grace 64 show");
        assert_eq!(ThinkConfig::default().to_string(), "off");
    }

    #[test]
    fn budget_counts_after_open_only() {
        let mut c = BudgetClock::new(4, 2);
        c.observe(true, false); // open tag
        assert_eq!(c.counted(), 0);
        assert_eq!(c.kind(true, false), ForceKind::Sample);
        c.observe(true, false); // body 1
        c.observe(true, false); // body 2 — bias_at = 2
        assert_eq!(c.counted(), 2);
        assert_eq!(c.kind(true, false), ForceKind::Bias);
        c.observe(true, false); // 3
        c.observe(true, false); // 4 — hard budget
        assert_eq!(c.counted(), 4);
        assert_eq!(c.kind(true, false), ForceKind::Force);
    }

    #[test]
    fn natural_close_never_forces() {
        let mut c = BudgetClock::new(8, 2);
        c.observe(true, false);
        c.observe(true, false);
        c.observe(false, false);
        assert_eq!(c.kind(false, false), ForceKind::Sample);
        c.observe(true, false); // new think-open re-arms
        assert_eq!(c.counted(), 0);
        assert_eq!(c.kind(true, false), ForceKind::Sample);
    }

    #[test]
    fn no_inject_while_partial_tag() {
        let mut c = BudgetClock::new(2, 0);
        c.observe(true, false); // open
        c.observe(true, false); // 1
        c.observe(true, true); // 2, still holding close prefix
        assert_eq!(c.counted(), 2);
        assert_eq!(c.kind(true, true), ForceKind::Bias);
        c.observe(true, false); // prefix resolved
        assert_eq!(c.kind(true, false), ForceKind::Force);
    }

    #[test]
    fn from_think_skips_unlimited() {
        assert!(BudgetClock::from_think(ThinkConfig::default(), 4096).is_none());
        let on = ThinkConfig {
            mode: ThinkMode::On,
            show: false,
        };
        assert!(BudgetClock::from_think(on, 4096).is_none());
        let budgeted = ThinkConfig {
            mode: ThinkMode::Budget {
                tokens: 256,
                grace: 32,
            },
            show: false,
        };
        let c = BudgetClock::from_think(budgeted, 4096).unwrap();
        assert_eq!(c.bias_at(), 224);
    }
}
