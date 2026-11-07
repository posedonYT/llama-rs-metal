mod benchmark;
mod inspect;
use clap::{Args, Parser, Subcommand};
use llama_rs::{
    Engine, EngineOptions, GenerationEvent, GenerationOptions, LoadOptions, Message, Sampling,
    SessionOptions,
};
use std::{
    io::{self, BufRead, Write},
    path::PathBuf,
    time::Instant,
};
#[derive(Parser)]
#[command(version, about = "Standalone Rust/Metal inference for Apple Silicon")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}
#[derive(Subcommand)]
enum Command {
    /// Inspect GGUF or a native MLX checkpoint without initializing the GPU.
    Inspect {
        model: PathBuf,
        #[arg(long)]
        tensors: bool,
    },
    /// Generate from a raw prompt (or format it as a chat message with --chat).
    Generate {
        #[command(flatten)]
        model: ModelArgs,
        #[arg(short, long)]
        prompt: String,
        #[arg(long)]
        chat: bool,
        #[command(flatten)]
        sampling: SampleArgs,
    },
    /// Interactive multi-turn chat. /reset clears history, /quit exits.
    Chat {
        #[command(flatten)]
        model: ModelArgs,
        #[arg(long)]
        system: Option<String>,
        #[command(flatten)]
        sampling: SampleArgs,
    },
    /// Fixed-token benchmark, JSON output suitable for reproducible comparisons.
    Bench {
        #[command(flatten)]
        model: ModelArgs,
        #[arg(long, default_value_t = 128)]
        prompt_tokens: usize,
        #[arg(long, default_value_t = 64)]
        decode_tokens: usize,
        #[arg(long, default_value_t = 3)]
        repetitions: usize,
        #[arg(long)]
        output: Option<PathBuf>,
    },
}
#[derive(Args)]
struct ModelArgs {
    /// GGUF file or local MLX checkpoint directory.
    #[arg(short, long)]
    model: PathBuf,
    #[arg(
        short = 'c',
        long,
        default_value_t = 0,
        help = "Context capacity (0 selects min(8192, model limit))"
    )]
    context: usize,
    #[arg(long)]
    memory_mib: Option<usize>,
    /// Cast MLX floating weights/scales to F32; preserve packed integer codes.
    #[arg(long)]
    float_weights_f32: bool,
}
#[derive(Args)]
struct SampleArgs {
    #[arg(short = 'n', long, default_value_t = 256)]
    max_tokens: usize,
    #[arg(long, default_value_t = 0.8)]
    temperature: f32,
    #[arg(long, default_value_t = 40)]
    top_k: usize,
    #[arg(long, default_value_t = 0.95)]
    top_p: f32,
    #[arg(long, default_value_t = 1.0)]
    repeat_penalty: f32,
    #[arg(long, default_value_t = 42)]
    seed: u64,
}
impl SampleArgs {
    fn options(&self) -> GenerationOptions {
        GenerationOptions {
            max_tokens: self.max_tokens,
            sampling: Sampling {
                temperature: self.temperature,
                top_k: self.top_k,
                top_p: self.top_p,
                repeat_penalty: self.repeat_penalty,
                seed: self.seed,
                ..Default::default()
            },
        }
    }
}
fn load(
    args: &ModelArgs,
) -> Result<(Engine, llama_rs::Model, llama_rs::Session, f64), Box<dyn std::error::Error>> {
    let start = Instant::now();
    let limit = args
        .memory_mib
        .map(|v| v.checked_mul(1024 * 1024).ok_or("memory limit overflow"))
        .transpose()?;
    let engine = Engine::new(EngineOptions {
        memory_limit: limit,
    })?;
    let model = engine.load_with_options(
        &args.model,
        LoadOptions {
            float_weights_f32: args.float_weights_f32,
        },
    )?;
    let session = model.session(SessionOptions {
        context: args.context,
    })?;
    eprintln!(
        "{}; {}; context {}; reserved model/session {:.1} MiB",
        engine.device_name(),
        model.config().architecture,
        session.context(),
        (model.memory(session.context())?.weights + model.memory(session.context())?.session)
            as f64
            / 1048576.
    );
    Ok((engine, model, session, start.elapsed().as_secs_f64()))
}
fn stream(
    session: &mut llama_rs::Session,
    prompt: &str,
    options: GenerationOptions,
) -> llama_rs::Result<String> {
    let mut text = String::new();
    for event in session.generate(prompt, options)? {
        let chunk = match event? {
            GenerationEvent::Token { text, .. } => text,
            GenerationEvent::Finished { text, .. } => text,
        };
        print!("{chunk}");
        let _ = io::stdout().flush();
        text.push_str(&chunk)
    }
    println!();
    Ok(text)
}
fn main() {
    if let Err(e) = run() {
        eprintln!("error: {e}");
        std::process::exit(1)
    }
}
fn run() -> Result<(), Box<dyn std::error::Error>> {
    match Cli::parse().command {
        Command::Inspect { model, tensors } => {
            inspect::run(&model, tensors)?;
        }
        Command::Generate {
            model: args,
            prompt,
            chat,
            sampling,
        } => {
            let (_engine, model, mut session, _) = load(&args)?;
            let prompt = if chat {
                model.render_chat(&[Message::new("user", prompt)], true)?
            } else {
                prompt
            };
            stream(&mut session, &prompt, sampling.options())?;
        }
        Command::Chat {
            model: args,
            system,
            sampling,
        } => {
            let (_engine, model, mut session, _) = load(&args)?;
            let mut messages = Vec::new();
            if let Some(s) = system {
                messages.push(Message::new("system", s));
            }
            let initial = messages.clone();
            let stdin = io::stdin();
            let mut lines = stdin.lock().lines();
            loop {
                print!("\n> ");
                io::stdout().flush()?;
                let Some(line) = lines.next() else { break };
                let line = line?;
                match line.trim() {
                    "/quit" | "/exit" => break,
                    "/reset" => {
                        session.reset()?;
                        messages = initial.clone();
                        continue;
                    }
                    "" => continue,
                    _ => {}
                }
                messages.push(Message::new("user", line));
                let prompt = model.render_chat(&messages, true)?;
                let response = stream(&mut session, &prompt, sampling.options())?;
                messages.push(Message::new("assistant", response));
            }
        }
        Command::Bench {
            model: args,
            prompt_tokens,
            decode_tokens,
            repetitions,
            output,
        } => {
            benchmark::run(&args, prompt_tokens, decode_tokens, repetitions, output)?;
        }
    }
    Ok(())
}
