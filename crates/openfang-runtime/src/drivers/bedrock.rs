use crate::llm_driver::{CompletionRequest, CompletionResponse, LlmDriver, LlmError};
use async_trait::async_trait;
use aws_sdk_bedrockruntime::types::{
    CachePointBlock, CachePointType, ContentBlock as BContentBlock, ConversationRole,
    InferenceConfiguration, Message as BMessage, SystemContentBlock, Tool, ToolConfiguration,
    ToolInputSchema, ToolResultBlock, ToolResultContentBlock, ToolSpecification, ToolUseBlock,
};
use openfang_types::message::{ContentBlock, MessageContent, Role, StopReason, TokenUsage};
use openfang_types::tool::{ToolCall, ToolDefinition};
use tracing::{debug, warn};

pub struct BedrockDriver;

impl BedrockDriver {
    pub fn new() -> Self {
        Self
    }
}

impl Default for BedrockDriver {
    fn default() -> Self {
        Self::new()
    }
}

fn supports_prompt_caching(model_id: &str) -> bool {
    let id = model_id.to_lowercase();
    id.contains("claude-haiku-4")
        || id.contains("claude-sonnet-4")
        || id.contains("claude-opus-4")
        || id.contains("nova-lite")
        || id.contains("nova-micro")
        || id.contains("nova-pro")
        || id.contains("nova-2-lite")
}

fn make_cache_point() -> CachePointBlock {
    CachePointBlock::builder()
        .r#type(CachePointType::Default)
        .build()
        .expect("CachePointBlock requires type")
}

fn json_to_document(value: &serde_json::Value) -> aws_smithy_types::Document {
    use aws_smithy_types::{Document, Number as SmithyNumber};
    use std::collections::HashMap;
    match value {
        serde_json::Value::Null => Document::Null,
        serde_json::Value::Bool(b) => Document::Bool(*b),
        serde_json::Value::String(s) => Document::String(s.clone()),
        serde_json::Value::Number(n) => {
            if let Some(u) = n.as_u64() {
                Document::Number(SmithyNumber::PosInt(u))
            } else if let Some(i) = n.as_i64() {
                Document::Number(SmithyNumber::NegInt(i))
            } else {
                Document::Number(SmithyNumber::Float(n.as_f64().unwrap_or(0.0)))
            }
        }
        serde_json::Value::Array(arr) => Document::Array(arr.iter().map(json_to_document).collect()),
        serde_json::Value::Object(map) => {
            let mut hmap = HashMap::new();
            for (k, v) in map {
                hmap.insert(k.clone(), json_to_document(v));
            }
            Document::Object(hmap)
        }
    }
}

fn document_to_json(doc: &aws_smithy_types::Document) -> serde_json::Value {
    use aws_smithy_types::Document;
    match doc {
        Document::Null => serde_json::Value::Null,
        Document::Bool(b) => serde_json::Value::Bool(*b),
        Document::String(s) => serde_json::Value::String(s.clone()),
        Document::Number(n) => serde_json::json!(n.to_f64_lossy()),
        Document::Array(arr) => serde_json::Value::Array(arr.iter().map(document_to_json).collect()),
        Document::Object(map) => {
            let obj: serde_json::Map<_, _> =
                map.iter().map(|(k, v)| (k.clone(), document_to_json(v))).collect();
            serde_json::Value::Object(obj)
        }
        _ => serde_json::Value::Null,
    }
}

fn build_bedrock_tools(tools: &[ToolDefinition]) -> Option<ToolConfiguration> {
    if tools.is_empty() {
        return None;
    }
    let mut bedrock_tools: Vec<Tool> = Vec::new();
    for tool in tools {
        match ToolSpecification::builder()
            .name(&tool.name)
            .description(&tool.description)
            .input_schema(ToolInputSchema::Json(json_to_document(&tool.input_schema)))
            .build()
        {
            Ok(s) => bedrock_tools.push(Tool::ToolSpec(s)),
            Err(e) => warn!(tool = %tool.name, error = %e, "Skipping tool spec build failure"),
        }
    }
    if bedrock_tools.is_empty() {
        return None;
    }
    let mut config = ToolConfiguration::builder();
    for t in bedrock_tools {
        config = config.tools(t);
    }
    config.build().ok()
}

#[async_trait]
impl LlmDriver for BedrockDriver {
    async fn complete(&self, request: CompletionRequest) -> Result<CompletionResponse, LlmError> {
        let aws_config = aws_config::from_env().load().await;
        let client = aws_sdk_bedrockruntime::Client::new(&aws_config);

        let caching = supports_prompt_caching(&request.model);
        debug!(model = %request.model, caching, tools = request.tools.len(), "Bedrock Converse");

        let system: Vec<SystemContentBlock> = match request.system.as_deref().filter(|s| !s.is_empty()) {
            None => vec![],
            Some(s) => {
                let mut blocks = vec![SystemContentBlock::Text(s.to_string())];
                if caching {
                    blocks.push(SystemContentBlock::CachePoint(make_cache_point()));
                }
                blocks
            }
        };

        let non_system: Vec<_> = request.messages.iter().filter(|m| m.role != Role::System).collect();
        let mut messages: Vec<BMessage> = Vec::new();

        for (i, msg) in non_system.iter().enumerate() {
            let role = match msg.role {
                Role::User => ConversationRole::User,
                Role::Assistant => ConversationRole::Assistant,
                Role::System => continue,
            };

            let is_cache_point_msg = caching
                && non_system.len() >= 3
                && i == non_system.len().saturating_sub(2);

            let mut builder = BMessage::builder().role(role);

            match &msg.content {
                MessageContent::Text(t) => {
                    builder = builder.content(BContentBlock::Text(t.clone()));
                    if is_cache_point_msg {
                        builder = builder.content(BContentBlock::CachePoint(make_cache_point()));
                    }
                }
                MessageContent::Blocks(blocks) => {
                    for (bi, block) in blocks.iter().enumerate() {
                        let is_last = bi == blocks.len() - 1;
                        match block {
                            ContentBlock::Text { text, .. } => {
                                builder = builder.content(BContentBlock::Text(text.clone()));
                                if is_cache_point_msg && is_last {
                                    builder = builder.content(BContentBlock::CachePoint(make_cache_point()));
                                }
                            }
                            ContentBlock::ToolUse { id, name, input, .. } => {
                                if let Ok(tu) = ToolUseBlock::builder()
                                    .tool_use_id(id)
                                    .name(name)
                                    .input(json_to_document(input))
                                    .build()
                                {
                                    builder = builder.content(BContentBlock::ToolUse(tu));
                                }
                            }
                            ContentBlock::ToolResult { tool_use_id, content, is_error, .. } => {
                                if let Ok(tr) = ToolResultBlock::builder()
                                    .tool_use_id(tool_use_id)
                                    .content(ToolResultContentBlock::Text(content.clone()))
                                    .set_status(if *is_error {
                                        Some(aws_sdk_bedrockruntime::types::ToolResultStatus::Error)
                                    } else {
                                        None
                                    })
                                    .build()
                                {
                                    builder = builder.content(BContentBlock::ToolResult(tr));
                                }
                            }
                            _ => {}
                        }
                    }
                }
            }

            match builder.build() {
                Ok(m) => messages.push(m),
                Err(e) => warn!(error = %e, "Skipping message build failure"),
            }
        }

        let inference = InferenceConfiguration::builder()
            .max_tokens(request.max_tokens as i32)
            .temperature(request.temperature)
            .build();

        let mut req = client
            .converse()
            .model_id(&request.model)
            .set_messages(Some(messages))
            .inference_config(inference);

        for block in system {
            req = req.system(block);
        }

        if let Some(tool_config) = build_bedrock_tools(&request.tools) {
            debug!(tool_count = request.tools.len(), "Attaching tool_config to Bedrock request");
            req = req.tool_config(tool_config);
        }

        let output = req.send().await.map_err(|e| {
            let msg = e.to_string();
            if msg.contains("ResourceNotFoundException") || msg.contains("ValidationException") {
                LlmError::ModelNotFound(request.model.clone())
            } else if msg.contains("AccessDeniedException") {
                LlmError::AuthenticationFailed(msg)
            } else if msg.contains("ThrottlingException") {
                LlmError::RateLimited { retry_after_ms: 5000 }
            } else {
                LlmError::Api { status: 0, message: msg }
            }
        })?;

        let stop_reason = match output.stop_reason().as_str() {
            "tool_use" => StopReason::ToolUse,
            "max_tokens" => StopReason::MaxTokens,
            _ => StopReason::EndTurn,
        };

        let usage = output
            .usage()
            .map(|u| {
                let cache_read = u.cache_read_input_tokens().unwrap_or(0) as u64;
                let cache_write = u.cache_write_input_tokens().unwrap_or(0) as u64;
                if caching && (cache_read > 0 || cache_write > 0) {
                    debug!(cache_read, cache_write, "Bedrock prompt cache hit");
                }
                TokenUsage {
                    input_tokens: u.input_tokens() as u64 + cache_read + cache_write,
                    output_tokens: u.output_tokens() as u64,
                }
            })
            .unwrap_or_default();

        let response_blocks = output
            .output()
            .and_then(|o| o.as_message().ok())
            .map(|m| m.content())
            .unwrap_or(&[]);

        let mut content: Vec<ContentBlock> = Vec::new();
        let mut tool_calls: Vec<ToolCall> = Vec::new();

        for block in response_blocks {
            match block {
                BContentBlock::Text(text) if !text.is_empty() => {
                    content.push(ContentBlock::Text { text: text.clone(), provider_metadata: None });
                }
                BContentBlock::ToolUse(tu) => {
                    let input = document_to_json(tu.input());
                    tool_calls.push(ToolCall {
                        id: tu.tool_use_id().to_string(),
                        name: tu.name().to_string(),
                        input: input.clone(),
                    });
                    content.push(ContentBlock::ToolUse {
                        id: tu.tool_use_id().to_string(),
                        name: tu.name().to_string(),
                        input,
                        provider_metadata: None,
                    });
                }
                _ => {}
            }
        }

        if content.is_empty() && tool_calls.is_empty() {
            content.push(ContentBlock::Text { text: String::new(), provider_metadata: None });
        }

        debug!(stop_reason = ?stop_reason, tool_calls = tool_calls.len(), "Bedrock response");

        Ok(CompletionResponse { content, stop_reason, tool_calls, usage })
    }
}
