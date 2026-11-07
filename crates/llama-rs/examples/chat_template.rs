fn main() -> Result<(), Box<dyn std::error::Error>> {
    let path = std::env::args().nth(1).ok_or("model path required")?;
    let f = model_format::Checkpoint::open(path)?;
    println!(
        "{}",
        f.chat_template.as_deref().unwrap_or("(built-in template)")
    );
    Ok(())
}
