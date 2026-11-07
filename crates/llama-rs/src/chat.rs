//! Jinja rendering policy is separate from model weights and session execution.
use crate::{Error, Message, Result};

pub(crate) fn render(
    source: &str,
    messages: &[Message],
    add_generation_prompt: bool,
    bos: &str,
    eos: &str,
) -> Result<String> {
    let mut env = minijinja::Environment::new();
    env.set_unknown_method_callback(minijinja_contrib::pycompat::unknown_method_callback);
    env.add_function(
        "raise_exception",
        |s: String| -> std::result::Result<String, minijinja::Error> {
            Err(minijinja::Error::new(
                minijinja::ErrorKind::InvalidOperation,
                s,
            ))
        },
    );
    env.add_template("chat", source)
        .map_err(|e| Error::Template(e.to_string()))?;
    let context = minijinja::context! {
        messages => messages,
        add_generation_prompt => add_generation_prompt,
        bos_token => bos,
        eos_token => eos,
        enable_thinking => false,
        // An empty list is still "not none" in Jinja. Several Llama templates
        // enable function-call instructions in that case, even with no tools.
        tools => Option::<Vec<String>>::None,
    };
    env.get_template("chat")
        .map_err(|e| Error::Template(e.to_string()))?
        .render(context)
        .map_err(|e| Error::Template(e.to_string()))
}

pub(crate) fn render_llama2(messages: &[Message], generation: bool) -> Result<String> {
    let mut out = String::new();
    let mut system = String::new();
    let mut expect_user = true;
    for (i, m) in messages.iter().enumerate() {
        match m.role.as_str() {
            "system" if i == 0 => system = m.content.clone(),
            "user" if expect_user => {
                out.push_str("<s>[INST] ");
                if !system.is_empty() {
                    out.push_str("<<SYS>>\n");
                    out.push_str(&system);
                    out.push_str("\n<</SYS>>\n\n");
                    system.clear()
                }
                out.push_str(&m.content);
                out.push_str(" [/INST]");
                expect_user = false
            }
            "assistant" if !expect_user => {
                out.push(' ');
                out.push_str(&m.content);
                out.push_str(" </s>");
                expect_user = true
            }
            _ => {
                return Err(Error::Template(
                    "Llama2 expects optional system followed by alternating user/assistant".into(),
                ));
            }
        }
    }
    if generation && expect_user {
        return Err(Error::Template(
            "generation requires a final user message".into(),
        ));
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn no_tools_uses_plain_chat_and_python_string_methods() {
        let source = "{% if tools is not none %}FUNCTION_CALL{% else %}{{messages[0].content.strip()}}{% endif %}";
        assert_eq!(
            render(source, &[Message::new("user", " hello ")], true, "", "").unwrap(),
            "hello"
        );
    }
    #[test]
    fn unsupported_template_function_is_an_error() {
        assert!(render("{{ unknown_template_function() }}", &[], true, "", "").is_err());
    }
}
