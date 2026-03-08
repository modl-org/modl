/// Detected GPU information
#[derive(Debug, Clone)]
pub struct GpuInfo {
    pub name: String,
    pub vram_mb: u64,
}

/// Detect GPU and VRAM. Returns None if no GPU found.
pub fn detect() -> Option<GpuInfo> {
    detect_nvml().or_else(detect_nvidia_smi)
}

/// Try NVML (programmatic nvidia-smi)
fn detect_nvml() -> Option<GpuInfo> {
    let nvml = nvml_wrapper::Nvml::init().ok()?;
    let device = nvml.device_by_index(0).ok()?;
    let name = device.name().ok()?;
    let memory = device.memory_info().ok()?;
    Some(GpuInfo {
        name,
        vram_mb: memory.total / (1024 * 1024),
    })
}

/// Fallback: parse nvidia-smi CLI output
fn detect_nvidia_smi() -> Option<GpuInfo> {
    let output = std::process::Command::new("nvidia-smi")
        .args([
            "--query-gpu=name,memory.total",
            "--format=csv,noheader,nounits",
        ])
        .output()
        .ok()?;

    if !output.status.success() {
        return None;
    }

    let stdout = String::from_utf8_lossy(&output.stdout);
    let line = stdout.lines().next()?;
    let parts: Vec<&str> = line.split(", ").collect();
    if parts.len() != 2 {
        return None;
    }

    let name = parts[0].trim().to_string();
    let vram_mb: u64 = parts[1].trim().parse().ok()?;

    Some(GpuInfo { name, vram_mb })
}

/// Select best variant based on available VRAM
pub fn select_variant(vram_mb: u64, variants: &[(String, u64)]) -> Option<String> {
    // variants: [(variant_id, vram_required_mb)]
    // Pick the largest variant that fits in VRAM
    let mut candidates: Vec<_> = variants
        .iter()
        .filter(|(_, required)| *required <= vram_mb)
        .collect();

    // Sort descending by VRAM requirement (prefer highest quality that fits)
    candidates.sort_by(|a, b| b.1.cmp(&a.1));
    candidates.first().map(|(id, _)| id.clone())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_select_variant() {
        let variants = vec![
            ("fp16".to_string(), 24576),
            ("fp8".to_string(), 12288),
            ("gguf-q4".to_string(), 8192),
            ("gguf-q2".to_string(), 4096),
        ];

        assert_eq!(select_variant(24576, &variants), Some("fp16".to_string()));
        assert_eq!(select_variant(16000, &variants), Some("fp8".to_string()));
        assert_eq!(
            select_variant(10000, &variants),
            Some("gguf-q4".to_string())
        );
        assert_eq!(select_variant(6000, &variants), Some("gguf-q2".to_string()));
        assert_eq!(select_variant(2000, &variants), None);
    }
}
