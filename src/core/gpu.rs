/// The type of GPU/accelerator device detected.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DeviceType {
    Cuda,
    Mps,
}

impl std::fmt::Display for DeviceType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Cuda => write!(f, "cuda"),
            Self::Mps => write!(f, "mps"),
        }
    }
}

/// Detected GPU information
#[derive(Debug, Clone)]
pub struct GpuInfo {
    pub name: String,
    pub vram_mb: u64,
    pub device: DeviceType,
}

/// Detect GPU and VRAM. Returns None if no GPU found.
///
/// Tries NVML → nvidia-smi → Apple Silicon MPS, in order.
pub fn detect() -> Option<GpuInfo> {
    detect_nvml().or_else(detect_nvidia_smi).or_else(detect_mps)
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
        device: DeviceType::Cuda,
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

    Some(GpuInfo {
        name,
        vram_mb,
        device: DeviceType::Cuda,
    })
}

/// Detect Apple Silicon GPU via MPS (Metal Performance Shaders).
///
/// Returns unified memory as VRAM (macOS shares memory between CPU and GPU).
fn detect_mps() -> Option<GpuInfo> {
    // Only meaningful on macOS ARM
    if !cfg!(target_os = "macos") {
        return None;
    }

    // Check architecture: must be arm64 (Apple Silicon)
    let arch_output = std::process::Command::new("uname")
        .arg("-m")
        .output()
        .ok()?;
    let arch = String::from_utf8_lossy(&arch_output.stdout);
    if arch.trim() != "arm64" {
        return None;
    }

    // Get chip name (e.g. "Apple M2 Max")
    let chip_output = std::process::Command::new("sysctl")
        .arg("-n")
        .arg("machdep.cpu.brand_string")
        .output()
        .ok()?;
    let name = String::from_utf8_lossy(&chip_output.stdout)
        .trim()
        .to_string();

    // Get total unified memory in bytes
    let mem_output = std::process::Command::new("sysctl")
        .arg("-n")
        .arg("hw.memsize")
        .output()
        .ok()?;
    let mem_bytes: u64 = String::from_utf8_lossy(&mem_output.stdout)
        .trim()
        .parse()
        .ok()?;
    // Unified memory is shared with the OS. Report ~75% as effective VRAM
    // to avoid selecting variants that would leave no room for the system.
    let vram_mb = (mem_bytes / (1024 * 1024)) * 3 / 4;

    Some(GpuInfo {
        name,
        vram_mb,
        device: DeviceType::Mps,
    })
}

/// Return the device string for the Python worker env var MODL_DEVICE.
///
/// Returns "cuda", "mps", or "cpu" based on detected hardware.
pub fn detect_device_str() -> String {
    detect()
        .map(|g| g.device.to_string())
        .unwrap_or_else(|| "cpu".to_string())
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
