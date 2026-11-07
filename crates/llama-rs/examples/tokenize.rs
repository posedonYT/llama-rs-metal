fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<_> = std::env::args().collect();
    if args.len() != 3 {
        return Err("usage: tokenize MODEL TEXT".into());
    }
    let f = model_format::Checkpoint::open(&args[1])?;
    let t = llama_rs::Tokenizer::from_checkpoint(&f)?;
    for id in t.encode(&args[2], true, true)? {
        println!("{id}")
    }
    Ok(())
}
