// Copyright (c) 2026 Ivan Tugay
// SPDX-License-Identifier: GPL-3.0-or-later
// Licensed under GPL-3.0 or later; see https://www.gnu.org/licenses/gpl-3.0.html

//! Binary entry. The program lives in the library so serve and daemon can
//! be linked without this file.

fn main() {
    runa::run();
}
