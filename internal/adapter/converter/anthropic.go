package converter

import (
	"encoding/json"

	"ollama-proxy/internal/domain"
)

// ========== Anthropic 请求 DTO ==========

type AnthropicMetadata struct {
	UserID string `json:"user_id,omitempty"`
}

type AnthropicMessage struct {
	Role    string      `json:"role"`
	Content interface{} `json:"content"` // 字符串或 []ContentBlock
}

type AnthropicTool struct {
	Name        string          `json:"name"`
	Description string          `json:"description,omitempty"`
	InputSchema json.RawMessage `json:"input_schema,omitempty"`
}

type AnthropicToolChoice struct {
	Type string `json:"type"` // "auto", "any", "tool", "none"
	Name string `json:"name,omitempty"`
}

type AnthropicThinking struct {
	Type         string `json:"type"` // "enabled" 或 "disabled"
	BudgetTokens int    `json:"budget_tokens,omitempty"`
}

type AnthropicRequest struct {
	Model         string               `json:"model"`
	Messages      []AnthropicMessage   `json:"messages"`
	System        interface{}          `json:"system,omitempty"` // 字符串或 []ContentBlock
	MaxTokens     int                  `json:"max_tokens"`
	Temperature   *float64             `json:"temperature,omitempty"`
	TopP          *float64             `json:"top_p,omitempty"`
	TopK          *int                 `json:"top_k,omitempty"`
	StopSequences []string             `json:"stop_sequences,omitempty"`
	Stream        bool                 `json:"stream,omitempty"`
	Metadata      *AnthropicMetadata   `json:"metadata,omitempty"`
	Tools         []AnthropicTool      `json:"tools,omitempty"`
	ToolChoice    *AnthropicToolChoice `json:"tool_choice,omitempty"`
	Thinking      *AnthropicThinking   `json:"thinking,omitempty"`
}

// ========== Anthropic 响应 DTO ==========

// AnthropicContentBlock 是通用的响应内容块；构建响应时使用具体类型块
// （AnthropicTextBlock、AnthropicToolUseBlock、AnthropicThinkingBlock），
// 以确保必填字段始终被序列化。
type AnthropicContentBlock struct {
	Type  string          `json:"type"`
	Text  string          `json:"text,omitempty"`
	ID    string          `json:"id,omitempty"`
	Name  string          `json:"name,omitempty"`
	Input json.RawMessage `json:"input,omitempty"`
}

type AnthropicUsage struct {
	InputTokens  int `json:"input_tokens"`
	OutputTokens int `json:"output_tokens"`
}

type AnthropicResponse struct {
	ID           string         `json:"id"`
	Type         string         `json:"type"`
	Role         string         `json:"role"`
	Content      []interface{}  `json:"content"`
	Model        string         `json:"model"`
	StopReason   string         `json:"stop_reason"`
	StopSequence *string        `json:"stop_sequence"`
	Usage        AnthropicUsage `json:"usage"`
}

type AnthropicCountTokensResponse struct {
	InputTokens int `json:"input_tokens"`
}

// ========== Anthropic SSE 事件类型 ==========

type AnthropicPing struct {
	Type string `json:"type"`
}

type AnthropicMessageStart struct {
	Type    string            `json:"type"`
	Message AnthropicStartMsg `json:"message"`
}

type AnthropicStartMsg struct {
	ID           string                  `json:"id"`
	Type         string                  `json:"type"`
	Role         string                  `json:"role"`
	Content      []AnthropicContentBlock `json:"content"`
	Model        string                  `json:"model"`
	StopReason   *string                 `json:"stop_reason"`
	StopSequence *string                 `json:"stop_sequence"`
	Usage        AnthropicUsage          `json:"usage"`
}

type AnthropicContentBlockStart struct {
	Type         string      `json:"type"`
	Index        int         `json:"index"`
	ContentBlock interface{} `json:"content_block"`
}

type AnthropicTextBlock struct {
	Type string `json:"type"`
	Text string `json:"text"`
}

type AnthropicToolUseBlock struct {
	Type  string          `json:"type"`
	ID    string          `json:"id"`
	Name  string          `json:"name"`
	Input json.RawMessage `json:"input"`
}

type AnthropicThinkingBlock struct {
	Type      string `json:"type"` // "thinking"
	Thinking  string `json:"thinking"`
	Signature string `json:"signature"`
}

type AnthropicContentBlockDelta struct {
	Type  string      `json:"type"`
	Index int         `json:"index"`
	Delta interface{} `json:"delta"`
}

type AnthropicTextDelta struct {
	Type string `json:"type"`
	Text string `json:"text"`
}

type AnthropicInputJSONDelta struct {
	Type        string `json:"type"` // "input_json_delta"
	PartialJSON string `json:"partial_json"`
}

type AnthropicThinkingDelta struct {
	Type     string `json:"type"` // "thinking_delta"
	Thinking string `json:"thinking"`
}

type AnthropicContentBlockStop struct {
	Type  string `json:"type"`
	Index int    `json:"index"`
}

type AnthropicMessageDelta struct {
	Type  string          `json:"type"`
	Delta AnthropicDelta  `json:"delta"`
	Usage *AnthropicUsage `json:"usage"`
}

type AnthropicDelta struct {
	StopReason   string  `json:"stop_reason"`
	StopSequence *string `json:"stop_sequence"`
}

type AnthropicMessageStop struct {
	Type string `json:"type"`
}

type AnthropicErr struct {
	Type    string `json:"type"`
	Message string `json:"message"`
}

type AnthropicErrorResp struct {
	Type  string       `json:"type"`
	Error AnthropicErr `json:"error"`
}

// ========== 转换函数 ==========

// AnthropicRequestToDomain 将 Anthropic 请求转换为领域 ChatRequest。
// 单个 Anthropic 消息可能扩展为多条领域消息，因为 tool_result 块映射到独立的 "tool" 角色消息。
func AnthropicRequestToDomain(apiReq AnthropicRequest) domain.ChatRequest {
	messages := make([]domain.Message, 0, len(apiReq.Messages)+1)

	if system := NormalizeContent(apiReq.System); system != "" {
		messages = append(messages, domain.Message{
			Role:    "system",
			Content: system,
		})
	}

	for _, msg := range apiReq.Messages {
		messages = append(messages, anthropicMessageToDomain(msg)...)
	}

	maxTokens := apiReq.MaxTokens
	req := domain.ChatRequest{
		Model:       apiReq.Model,
		Messages:    messages,
		Stream:      apiReq.Stream,
		Temperature: apiReq.Temperature,
		TopP:        apiReq.TopP,
		TopK:        apiReq.TopK,
		MaxTokens:   &maxTokens,
		Stop:        apiReq.StopSequences,
	}

	if apiReq.Thinking != nil {
		think := apiReq.Thinking.Type == "enabled"
		req.Think = &think
	}

	// tool_choice "none" 禁用工具；Ollama 无法强制使用特定工具，
	// 因此其他选择按原样传递工具。
	if apiReq.ToolChoice != nil && apiReq.ToolChoice.Type == "none" {
		return req
	}
	for _, t := range apiReq.Tools {
		req.Tools = append(req.Tools, domain.Tool{
			Name:        t.Name,
			Description: t.Description,
			Parameters:  t.InputSchema,
		})
	}
	return req
}

// anthropicMessageToDomain 将一条 Anthropic 消息展开为领域消息。
func anthropicMessageToDomain(msg AnthropicMessage) []domain.Message {
	// 纯字符串内容。
	if text, ok := msg.Content.(string); ok {
		return []domain.Message{{Role: msg.Role, Content: text}}
	}

	blocks, ok := msg.Content.([]interface{})
	if !ok {
		return []domain.Message{{Role: msg.Role, Content: NormalizeContent(msg.Content)}}
	}

	var result []domain.Message
	current := domain.Message{Role: msg.Role}
	hasCurrent := false

	flush := func() {
		if hasCurrent {
			result = append(result, current)
			current = domain.Message{Role: msg.Role}
			hasCurrent = false
		}
	}

	for _, raw := range blocks {
		block, ok := raw.(map[string]interface{})
		if !ok {
			continue
		}
		switch block["type"] {
		case "text":
			if txt, ok := block["text"].(string); ok {
				current.Content += txt
				hasCurrent = true
			}
		case "thinking":
			if txt, ok := block["thinking"].(string); ok {
				current.Thinking += txt
				hasCurrent = true
			}
		case "image":
			if src, ok := block["source"].(map[string]interface{}); ok {
				switch src["type"] {
				case "url":
					// 保留为 URL；处理器会在请求到达 Ollama 之前下载它。
					if url, ok := src["url"].(string); ok && IsRemoteURL(url) {
						current.Images = append(current.Images, url)
						hasCurrent = true
					}
				default: // "base64"
					if data, ok := src["data"].(string); ok && data != "" {
						current.Images = append(current.Images, data)
						hasCurrent = true
					}
				}
			}
		case "tool_use":
			id, _ := block["id"].(string)
			name, _ := block["name"].(string)
			args := "{}"
			if input, exists := block["input"]; exists && input != nil {
				if encoded, err := json.Marshal(input); err == nil {
					args = string(encoded)
				}
			}
			current.ToolCalls = append(current.ToolCalls, domain.ToolCall{
				ID:        id,
				Name:      name,
				Arguments: args,
			})
			hasCurrent = true
		case "tool_result":
			// 工具结果变为独立的 "tool" 角色消息。
			flush()
			toolUseID, _ := block["tool_use_id"].(string)
			result = append(result, domain.Message{
				Role:       "tool",
				Content:    normalizeToolResultContent(block["content"]),
				ToolCallID: toolUseID,
			})
		}
	}
	flush()

	return result
}

// normalizeToolResultContent 展平 tool_result 内容（字符串或块数组）。
func normalizeToolResultContent(content interface{}) string {
	switch v := content.(type) {
	case string:
		return v
	case []interface{}:
		text := ""
		for _, raw := range v {
			if block, ok := raw.(map[string]interface{}); ok && block["type"] == "text" {
				if txt, ok := block["text"].(string); ok {
					text += txt
				}
			}
		}
		return text
	default:
		return ""
	}
}

// DomainToAnthropicResponse 将领域 ChatResponse 转换为 Anthropic 响应。
func DomainToAnthropicResponse(model string, resp *domain.ChatResponse) AnthropicResponse {
	var content []interface{}
	if resp.Thinking != "" {
		content = append(content, AnthropicThinkingBlock{
			Type:     "thinking",
			Thinking: resp.Thinking,
		})
	}
	if resp.Content != "" || (len(resp.ToolCalls) == 0 && resp.Thinking == "") {
		content = append(content, AnthropicTextBlock{Type: "text", Text: resp.Content})
	}
	for _, tc := range resp.ToolCalls {
		input := json.RawMessage(tc.Arguments)
		if !json.Valid(input) {
			input = json.RawMessage("{}")
		}
		content = append(content, AnthropicToolUseBlock{
			Type:  "tool_use",
			ID:    tc.ID,
			Name:  tc.Name,
			Input: input,
		})
	}

	stopReason := MapStopReason(resp.FinishReason)
	if len(resp.ToolCalls) > 0 {
		stopReason = "tool_use"
	}

	return AnthropicResponse{
		ID:           "msg_" + FormatHex(),
		Type:         "message",
		Role:         "assistant",
		Content:      content,
		Model:        model,
		StopReason:   stopReason,
		StopSequence: nil,
		Usage: AnthropicUsage{
			InputTokens:  resp.InputTokens,
			OutputTokens: resp.OutputTokens,
		},
	}
}

// EstimateAnthropicTokens 粗略统计 count_tokens 请求的输入 token 数。
func EstimateAnthropicTokens(apiReq AnthropicRequest) int {
	total := EstimateTokens(NormalizeContent(apiReq.System))
	for _, msg := range apiReq.Messages {
		for _, dm := range anthropicMessageToDomain(msg) {
			total += EstimateTokens(dm.Content)
			for _, tc := range dm.ToolCalls {
				total += EstimateTokens(tc.Arguments)
			}
		}
	}
	for _, t := range apiReq.Tools {
		total += EstimateTokens(t.Name + t.Description + string(t.InputSchema))
	}
	if total < 1 {
		total = 1
	}
	return total
}

// MapStopReason 将 Ollama 风格的停止原因转换为 Anthropic 风格的停止原因。
func MapStopReason(reason string) string {
	switch reason {
	case "length", "max_tokens":
		return "max_tokens"
	case "tool_calls":
		return "tool_use"
	default:
		return "end_turn"
	}
}
