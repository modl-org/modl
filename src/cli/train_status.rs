use anyhow::Result;
use console::style;

use crate::core::training_status;

/// Show live training status for all runs or a specific one.
pub fn run(name: Option<&str>, watch: bool) -> Result<()> {
    if watch {
        run_watch(name)
    } else {
        run_once(name)
    }
}

fn run_once(name: Option<&str>) -> Result<()> {
    let runs = if let Some(name) = name {
        vec![training_status::get_status(name)?]
    } else {
        training_status::get_all_status(false)?
    };

    if runs.is_empty() {
        println!(
            "{} No training runs found in ~/.modl/training_output/",
            style("·").dim()
        );
        return Ok(());
    }

    for run in &runs {
        print_run_status(run);
        println!();
    }

    Ok(())
}

fn run_watch(name: Option<&str>) -> Result<()> {
    loop {
        // Clear screen
        print!("\x1B[2J\x1B[1;1H");

        let runs = if let Some(name) = name {
            vec![training_status::get_status(name)?]
        } else {
            // In watch mode, only show active runs
            let all = training_status::get_all_status(true)?;
            if all.is_empty() {
                training_status::get_all_status(false)?
            } else {
                all
            }
        };

        println!("{}\n", style("modl train status").bold().underlined());

        if runs.is_empty() {
            println!("  {} No training runs found.", style("·").dim());
        } else {
            for run in &runs {
                print_run_status(run);
                println!();
            }
        }

        println!(
            "{}",
            style("  Refreshing every 2s · Press Ctrl+C to exit").dim()
        );

        std::thread::sleep(std::time::Duration::from_secs(2));
    }
}

fn print_run_status(run: &training_status::TrainingProgress) {
    // Header: name + status indicator
    let status_icon = if run.is_running {
        style("●").green().bold()
    } else if run.current_step.is_some() {
        style("○").dim()
    } else {
        style("·").dim()
    };

    let status_label = if run.is_running {
        style("training").green()
    } else if run.current_step == run.total_steps && run.total_steps.is_some() {
        style("completed").cyan()
    } else {
        style("stopped").dim()
    };

    println!(
        "  {} {} {}",
        status_icon,
        style(&run.name).bold(),
        status_label,
    );

    // Model info
    if let Some(ref arch) = run.arch {
        let model_str = if let Some(ref base) = run.base_model {
            format!("{arch} · {base}")
        } else {
            arch.clone()
        };
        println!("    Model:   {}", style(model_str).dim());
    }

    if let Some(ref trigger) = run.trigger_word {
        println!("    Trigger: {}", style(trigger).dim());
    }

    // Progress bar
    if let (Some(step), Some(total)) = (run.current_step, run.total_steps) {
        let pct = if total > 0 {
            (step as f32 / total as f32 * 100.0).min(100.0)
        } else {
            0.0
        };

        let bar_width = 30;
        let filled = (pct / 100.0 * bar_width as f32) as usize;
        let empty = bar_width - filled;
        let bar = format!("{}{}", "█".repeat(filled), "░".repeat(empty),);

        println!(
            "    Progress {} {}/{} ({:.1}%)",
            style(bar).cyan(),
            style(step).bold(),
            total,
            pct,
        );

        // Speed + timing
        let mut timing_parts = Vec::new();
        if let Some(ref elapsed) = run.elapsed {
            timing_parts.push(format!("elapsed: {elapsed}"));
        }
        if let Some(ref eta) = run.eta {
            timing_parts.push(format!("eta: {eta}"));
        }
        if let Some(speed) = run.speed {
            timing_parts.push(format!("{speed:.2} it/s"));
        }
        if !timing_parts.is_empty() {
            println!("    Timing:  {}", style(timing_parts.join(" · ")).dim());
        }

        // Loss + LR
        let mut train_parts = Vec::new();
        if let Some(loss) = run.loss {
            train_parts.push(format!("loss: {loss:.4}"));
        }
        if let Some(ref lr) = run.lr {
            train_parts.push(format!("lr: {lr}"));
        }
        if !train_parts.is_empty() {
            println!("    Train:   {}", style(train_parts.join(" · ")).dim());
        }
    } else {
        println!("    {}", style("No progress data available").dim());
    }

    // Latest checkpoint
    if let Some(ref ckpt) = run.latest_checkpoint {
        let short = ckpt.rsplit('/').next().unwrap_or(ckpt);
        println!("    Ckpt:    {}", style(short).dim());
    }

    // Samples count
    if !run.latest_samples.is_empty() {
        println!(
            "    Samples: {} latest images",
            style(run.latest_samples.len()).dim()
        );
    }
}
