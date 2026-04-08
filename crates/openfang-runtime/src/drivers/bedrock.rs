use crate::llm_driver::{CompletionRequest, CompletionResponse, LlmDriver, LlmError};
use async_trait::async_trait;
use aws_sdk_bedrockruntime::types::{
    ContentBlock as BContentBlock, ConversationRole, InferenceConfiguration,
    Message as BMessage, SystemContentBlock,
};
use openfang_types::message::{ContentBlock, MessageContent, Role, StopReason, TokenUsage};
use tracing::debug;

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

#[async_trait]
impl LlmDriver for BedrockDriver {
    async fn complete(&self, request: CompletionRequest) -> Result<CompletionResponse, LlmError> {
        let aws_config = aws_config::from_env().load().await;
        let client = aws_sdk_bedrockruntime::Client::new(&aws_config);

        let system: Vec<SystemContentBlock> = request
            .system
            .as_deref()
            .filter(|s| !s.is_empty())
            .into_iter()
            .map(|s| SystemContentBlock::Text(s.to_string()))
            .collect();

        let mut messages: Vec<BMessage> = Vec::new();
        for msg in &request.messages {
            if msg.role == Role::System {
                continue;
            }
            let role = match msg.role {
                Role::User => ConversationRole::User,
                Role::Assistant => ConversationRole::Assistant,
                Role::System => continue,
            };
            let text = match &msg.content {
                MessageContent::Text(t) => t.clone(),
                MessageContent::Blocks(blocks) => blocks
                    .iter()
                    .filter_map(|b| match b {
                        ContentBlock::Text { text, .. } => Some(text.as_str()),
                        _ => None,
                    })
                    .collect::<Vec<_>>()
                    .join(""),
            };
            let bedrock_msg = BMessage::builder()
                .role(role)
                .content(BContentBlock::Text(text))
                .build()
                .map_err(|e| LlmError::Http(e.to_string()))?;
            messages.push(bedrock_msg);
        }

        let inference = InferenceConfiguration::builder()
            .max_tokens(request.max_tokens as i32)
            .temperature(request.temperature)
            .build();

        debug!(model = %request.model, "Sending Bedrock Converse request");

        let mut req = client
            .converse()
            .model_id(&request.model)
            .set_messages(Some(messages))
            .inference_config(inference);

        for block in system {
            req = req.system(block);
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
            .map(|u| TokenUsage {
                input_tokens: u.input_tokens() as u64,
                output_tokens: u.output_tokens() as u64,
            })
            .unwrap_or_default();

        let text = output
            .output()
            .and_then(|o| o.as_message().ok())
            .map(|m| {
                m.content()
                    .iter()
                    .filter_map(|b| b.as_text().ok().map(|s| s.to_string()))
                    .collect::<Vec<_>>()
                    .join("")
            })
            .unwrap_or_default();

        Ok(CompletionResponse {
            content: vec![ContentBlock::Text {
                text,
                provider_metadata: None,
            }],
            stop_reason,
            tool_calls: vec![],
            usage,
        })
    }
}
