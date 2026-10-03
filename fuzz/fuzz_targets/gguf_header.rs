//! GGUF header bytes → `Reader::parse` → `Descriptor` → fit report.
//!
//! Mirrors what `runa fit <file.gguf>` does with the header prefix of a
//! local or remote model: every byte is attacker-controlled (downloaded
//! files, HTTP range responses).
#![no_main]

use libfuzzer_sys::fuzz_target;
use runa_fit::{
    Descriptor, FitConfig, HwSpec, Reader, check_fit, estimate_compute, estimate_kv,
    estimate_speed_single, format_report,
};

fuzz_target!(|data: &[u8]| {
    let Ok(reader) = Reader::parse(data) else {
        return;
    };
    // Typed getters over whatever keys the header declared.
    for key in ["general.architecture", "general.name", "general.file_type"] {
        let _ = reader.get(key);
        let _ = reader.get_str(key);
        let _ = reader.get_u32(key);
        let _ = reader.get_u64(key);
        let _ = reader.get_f32(key);
        let _ = reader.get_bool(key);
    }
    let Ok(desc) = Descriptor::from_reader(&reader) else {
        return;
    };
    let config = FitConfig::default();
    let kv = estimate_kv(&desc, config.planner.ctx_len, &config.planner.kv_type);
    let _ = estimate_compute(&desc, config.planner.n_ubatch);
    let _ = estimate_speed_single(&desc, &kv, config.planner.ctx_len, 512, &HwSpec::cpu());
    let report = check_fit(&desc, &config);
    let _ = format_report(&report);
});
