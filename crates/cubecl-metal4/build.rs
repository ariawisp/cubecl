use std::env;

fn parse_version(v: &str) -> Option<(u64, u64)> {
    let mut it = v.split('.');
    let major = it.next()?.trim().parse().ok()?;
    let minor = it.next().unwrap_or("0").trim().parse().ok()?;
    Some((major, minor))
}

fn main() {
    // Enforce platform at compile time: Metal 4 requires macOS target.
    let target_os = env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
    if target_os != "macos" {
        panic!(
            "cubecl-metal4 requires a macOS target (found target_os={}). Disable the 'metal4' feature or switch target.",
            target_os
        );
    }

    // Enforce a minimum deployment target when available. Metal 4 APIs are available on macOS 15+.
    if let Ok(dep) = env::var("MACOSX_DEPLOYMENT_TARGET") {
        if let Some((maj, _min)) = parse_version(&dep) {
            if maj < 15 {
                panic!(
                    "MACOSX_DEPLOYMENT_TARGET={} is too low for Metal 4. Set MACOSX_DEPLOYMENT_TARGET=15.0 or newer.",
                    dep
                );
            }
        }
    }
}

