package converter

import (
	"encoding/json"
	"fmt"
	"strings"

	"ollama-proxy/internal/domain"
)

// ========== OpenAI 请求 DTO ==========

type OpenAIToolCallFunction struct {
	Name      string `json:"name,omitempty"`
	Arguments string `json:"arguments,omitempty"`
}

type OpenAIToolCall struct {
	Index    *int                   `json:"index,omitempty"` // 在流式增量中存在
	ID       string                 `json:"id,omitempty"`
	Type     string                 `json:"type,omitempty"`
	Function OpenAIToolCallFunction `json:"function"`
}

type OpenAIMessage struct {
	Role             string           `json:"role"`
	Content          interface{}      `json:"content"` // 字符串或 []ContentPart
	ReasoningContent string           `json:"reasoning_content,omitempty"`
	Name             string           `json:"name,omitempty"`
	ToolCalls        []OpenAIToolCall `json:"tool_calls,omitempty"`
	ToolCallID       string           `json:"tool_call_id,omitempty"`
}

type OpenAIToolFunction struct {
	Name        string          `json:"name"`
	Description string          `json:"description,omitempty"`
	Parameters  json.RawMessage `json:"parameters,omitempty"`
}

type OpenAITool struct {
	Type     string             `json:"type"`
	Function OpenAIToolFunction `json:"function"`
}

type OpenAIStreamOptions struct {
	IncludeUsage bool `json:"include_usage"`
}

type OpenAIJSONSchemaSpec struct {
	Name   string          `json:"name,omitempty"`
	Schema json.RawMessage `json:"schema,omitempty"`
	Strict *bool           `json:"strict,omitempty"`
}

type OpenAIResponseFormat struct {
	Type       string                `json:"type"` // "text", "json_object", "json_schema"
	JSONSchema *OpenAIJSONSchemaSpec `json:"json_schema,omitempty"`
}

type OpenAIChatRequest struct {
	Model               string                `json:"model"`
	Messages            []OpenAIMessage       `json:"messages"`
	Stream              bool                  `json:"stream"`
	StreamOptions       *OpenAIStreamOptions  `json:"stream_options,omitempty"`
	Temperature         *float64              `json:"temperature,omitempty"`
	TopP                *float64              `json:"top_p,omitempty"`
	MaxTokens           *int                  `json:"max_tokens,omitempty"`
	MaxCompletionTokens *int                  `json:"max_completion_tokens,omitempty"`
	Stop                interface{}           `json:"stop,omitempty"` // 字符串或 []string
	Seed                *int                  `json:"seed,omitempty"`
	PresencePenalty     *float64              `json:"presence_penalty,omitempty"`
	FrequencyPenalty    *float64              `json:"frequency_penalty,omitempty"`
	Tools               []OpenAITool          `json:"tools,omitempty"`
	ToolChoice          interface{}           `json:"tool_choice,omitempty"` // "none"/"auto"/"required" 或对象
	ResponseFormat      *OpenAIResponseFormat `json:"response_format,omitempty"`
}

// ========== OpenAI 响应 DTO ==========

type OpenAIStreamDelta struct {
	Role             string           `json:"role,omitempty"`
	Content          string           `json:"content,omitempty"`
	ReasoningContent string           `json:"reasoning_content,omitempty"`
	ToolCalls        []OpenAIToolCall `json:"tool_calls,omitempty"`
}

type OpenAIStreamChoice struct {
	Index        int               `json:"index"`
	Delta        OpenAIStreamDelta `json:"delta"`
	FinishReason *string           `json:"finish_reason"`
}

type OpenAIChunk struct {
	ID                string               `json:"id"`
	Object            string               `json:"object"`
	Created           int64                `json:"created"`
	Model             string               `json:"model"`
	SystemFingerprint string               `json:"system_fingerprint"`
	Choices           []OpenAIStreamChoice `json:"choices"`
	Usage             *OpenAIUsage         `json:"usage,omitempty"`
}

type OpenAIRespMessage struct {
	Role             string           `json:"role"`
	Content          *string          `json:"content"`
	ReasoningContent string           `json:"reasoning_content,omitempty"`
	ToolCalls        []OpenAIToolCall `json:"tool_calls,omitempty"`
}

type OpenAIChoice struct {
	Index        int               `json:"index"`
	Message      OpenAIRespMessage `json:"message"`
	FinishReason string            `json:"finish_reason"`
}

type OpenAIUsage struct {
	PromptTokens     int `json:"prompt_tokens"`
	CompletionTokens int `json:"completion_tokens"`
	TotalTokens      int `json:"total_tokens"`
}

type OpenAIResponse struct {
	ID                string         `json:"id"`
	Object            string         `json:"object"`
	Created           int64          `json:"created"`
	Model             string         `json:"model"`
	SystemFingerprint string         `json:"system_fingerprint"`
	Choices           []OpenAIChoice `json:"choices"`
	Usage             *OpenAIUsage   `json:"usage,omitempty"`
}

type OpenAIError struct {
	Message string  `json:"message"`
	Type    string  `json:"type"`
	Param   *string `json:"param"`
	Code    *string `json:"code"`
}

type OpenAIErrorResponse struct {
	Error OpenAIError `json:"error"`
}

// ========== OpenAI 模型 DTO ==========

type OpenAIModel struct {
	ID      string `json:"id"`
	Object  string `json:"object"`
	Created int64  `json:"created"`
	OwnedBy string `json:"owned_by"`
}

type OpenAIModelList struct {
	Object string        `json:"object"`
	Data   []OpenAIModel `json:"data"`
}

// ========== OpenAI 旧版补全 DTO ==========

type OpenAICompletionRequest struct {
	Model            string               `json:"model"`
	Prompt           interface{}          `json:"prompt"` // 字符串或 []string
	Suffix           string               `json:"suffix,omitempty"`
	MaxTokens        *int                 `json:"max_tokens,omitempty"`
	Temperature      *float64             `json:"temperature,omitempty"`
	TopP             *float64             `json:"top_p,omitempty"`
	Stop             interface{}          `json:"stop,omitempty"`
	Seed             *int                 `json:"seed,omitempty"`
	PresencePenalty  *float64             `json:"presence_penalty,omitempty"`
	FrequencyPenalty *float64             `json:"frequency_penalty,omitempty"`
	Stream           bool                 `json:"stream"`
	StreamOptions    *OpenAIStreamOptions `json:"stream_options,omitempty"`
}

type OpenAICompletionChoice struct {
	Text         string      `json:"text"`
	Index        int         `json:"index"`
	Logprobs     interface{} `json:"logprobs"`
	FinishReason *string     `json:"finish_reason"`
}

type OpenAICompletionResponse struct {
	ID      string                   `json:"id"`
	Object  string                   `json:"object"`
	Created int64                    `json:"created"`
	Model   string                   `json:"model"`
	Choices []OpenAICompletionChoice `json:"choices"`
	Usage   *OpenAIUsage             `json:"usage,omitempty"`
}

// ========== OpenAI 嵌入向量 DTO ==========

type OpenAIEmbeddingRequest struct {
	Model string      `json:"model"`
	Input interface{} `json:"input"` // 字符串或 []string
}

type OpenAIEmbeddingData struct {
	Object    string    `json:"object"`
	Index     int       `json:"index"`
	Embedding []float64 `json:"embedding"`
}

type OpenAIEmbeddingResponse struct {
	Object string                `json:"object"`
	Data   []OpenAIEmbeddingData `json:"data"`
	Model  string                `json:"model"`
	Usage  OpenAIUsage           `json:"usage"`
}

const systemFingerprint = "fp_ollama"

// ========== 转换函数 ==========

// OpenAIRequestToDomain 将 OpenAI 请求转换为领域 ChatRequest。
func OpenAIRequestToDomain(apiReq OpenAIChatRequest) domain.ChatRequest {
	messages := make([]domain.Message, len(apiReq.Messages))
	for i, msg := range apiReq.Messages {
		text, images := NormalizeOpenAIContent(msg.Content)
		dm := domain.Message{
			Role:       msg.Role,
			Content:    text,
			Thinking:   msg.ReasoningContent,
			Images:     images,
			ToolCallID: msg.ToolCallID,
			ToolName:   msg.Name,
		}
		for _, tc := range msg.ToolCalls {
			dm.ToolCalls = append(dm.ToolCalls, domain.ToolCall{
				ID:        tc.ID,
				Name:      tc.Function.Name,
				Arguments: tc.Function.Arguments,
			})
		}
		messages[i] = dm
	}

	maxTokens := apiReq.MaxTokens
	if apiReq.MaxCompletionTokens != nil {
		maxTokens = apiReq.MaxCompletionTokens
	}

	req := domain.ChatRequest{
		Model:            apiReq.Model,
		Messages:         messages,
		Stream:           apiReq.Stream,
		Temperature:      apiReq.Temperature,
		TopP:             apiReq.TopP,
		MaxTokens:        maxTokens,
		Stop:             NormalizeStop(apiReq.Stop),
		Seed:             apiReq.Seed,
		PresencePenalty:  apiReq.PresencePenalty,
		FrequencyPenalty: apiReq.FrequencyPenalty,
		Format:           ResponseFormatToOllama(apiReq.ResponseFormat),
	}

	// tool_choice "none" 禁用工具；Ollama 无法强制使用特定工具，
	// 因此其他选择按原样传递工具。
	if choice, ok := apiReq.ToolChoice.(string); ok && choice == "none" {
		return req
	}
	for _, t := range apiReq.Tools {
		req.Tools = append(req.Tools, domain.Tool{
			Name:        t.Function.Name,
			Description: t.Function.Description,
			Parameters:  t.Function.Parameters,
		})
	}
	return req
}

// DomainToOpenAIResponse 将领域 ChatResponse 转换为 OpenAI 响应。
func DomainToOpenAIResponse(model string, resp *domain.ChatResponse) OpenAIResponse {
	msg := OpenAIRespMessage{
		Role:             "assistant",
		ReasoningContent: resp.Thinking,
		ToolCalls:        toOpenAIToolCalls(resp.ToolCalls, false),
	}
	if resp.Content != "" || len(resp.ToolCalls) == 0 {
		content := resp.Content
		msg.Content = &content
	}

	return OpenAIResponse{
		ID:                fmt.Sprintf("chatcmpl-%d", IDTimestamp()),
		Object:            "chat.completion",
		Created:           NowUnix(),
		Model:             model,
		SystemFingerprint: systemFingerprint,
		Choices: []OpenAIChoice{
			{
				Index:        0,
				Message:      msg,
				FinishReason: MapOpenAIFinishReason(resp.FinishReason, len(resp.ToolCalls) > 0),
			},
		},
		Usage: newOpenAIUsage(resp.InputTokens, resp.OutputTokens),
	}
}

// BuildOpenAIStreamChunk 从领域 StreamChunk 构建 OpenAI SSE 块。
// toolCallIndex 跟踪整个流中的工具调用索引。
// 当块不包含任何可发送内容时返回 false。
func BuildOpenAIStreamChunk(id string, created int64, model string, chunk domain.StreamChunk, toolCallIndex *int, sawToolCalls bool) (OpenAIChunk, bool) {
	base := OpenAIChunk{
		ID:                id,
		Object:            "chat.completion.chunk",
		Created:           created,
		Model:             model,
		SystemFingerprint: systemFingerprint,
	}

	if chunk.Done {
		finishReason := "stop"
		if chunk.FinishReason != nil {
			finishReason = *chunk.FinishReason
		}
		finishReason = MapOpenAIFinishReason(finishReason, sawToolCalls)
		base.Choices = []OpenAIStreamChoice{
			{Index: 0, Delta: OpenAIStreamDelta{}, FinishReason: &finishReason},
		}
		return base, true
	}

	delta := OpenAIStreamDelta{Role: "assistant"}
	if chunk.Content != "" {
		delta.Content = chunk.Content
	}
	if chunk.Thinking != "" {
		delta.ReasoningContent = chunk.Thinking
	}
	if len(chunk.ToolCalls) > 0 {
		calls := toOpenAIToolCalls(chunk.ToolCalls, true)
		for i := range calls {
			idx := *toolCallIndex
			calls[i].Index = &idx
			*toolCallIndex++
		}
		delta.ToolCalls = calls
	}

	if delta.Content == "" && delta.ReasoningContent == "" && len(delta.ToolCalls) == 0 {
		return OpenAIChunk{}, false
	}

	base.Choices = []OpenAIStreamChoice{
		{Index: 0, Delta: delta, FinishReason: nil},
	}
	return base, true
}

// BuildOpenAIUsageChunk 创建仅包含用量信息的 SSE 块。
func BuildOpenAIUsageChunk(id string, created int64, model string, inputTokens, outputTokens int) OpenAIChunk {
	return OpenAIChunk{
		ID:                id,
		Object:            "chat.completion.chunk",
		Created:           created,
		Model:             model,
		SystemFingerprint: systemFingerprint,
		Choices:           []OpenAIStreamChoice{},
		Usage:             newOpenAIUsage(inputTokens, outputTokens),
	}
}

// DomainToOpenAIModelList 将模型信息转换为 OpenAI 模型列表格式。
func DomainToOpenAIModelList(models []domain.ModelInfo) OpenAIModelList {
	data := make([]OpenAIModel, len(models))
	for i, m := range models {
		data[i] = OpenAIModel{
			ID:      m.Name,
			Object:  "model",
			Created: NowUnix(),
			OwnedBy: "ollama",
		}
	}
	return OpenAIModelList{Object: "list", Data: data}
}

// DomainToOpenAIEmbeddings 将嵌入向量结果转换为 OpenAI 格式。
func DomainToOpenAIEmbeddings(model string, result *domain.EmbeddingResult) OpenAIEmbeddingResponse {
	data := make([]OpenAIEmbeddingData, len(result.Embeddings))
	for i, emb := range result.Embeddings {
		data[i] = OpenAIEmbeddingData{
			Object:    "embedding",
			Index:     i,
			Embedding: emb,
		}
	}
	return OpenAIEmbeddingResponse{
		Object: "list",
		Data:   data,
		Model:  model,
		Usage: OpenAIUsage{
			PromptTokens: result.InputTokens,
			TotalTokens:  result.InputTokens,
		},
	}
}

// ResponseFormatToOllama 将 OpenAI response_format 映射为 Ollama format 参数：
// "json" 对应 json_object，原始 schema 对应 json_schema。
func ResponseFormatToOllama(rf *OpenAIResponseFormat) json.RawMessage {
	if rf == nil {
		return nil
	}
	switch rf.Type {
	case "json_object":
		return json.RawMessage(`"json"`)
	case "json_schema":
		if rf.JSONSchema != nil && len(rf.JSONSchema.Schema) > 0 {
			return rf.JSONSchema.Schema
		}
		return json.RawMessage(`"json"`)
	default:
		return nil
	}
}

// CompletionRequestToDomain 将旧版补全请求转换为领域 GenerateRequest。
// 当 prompt 不是单个字符串时返回 ok=false。
func CompletionRequestToDomain(apiReq OpenAICompletionRequest) (domain.GenerateRequest, bool) {
	prompt := ""
	switch v := apiReq.Prompt.(type) {
	case string:
		prompt = v
	case []interface{}:
		if len(v) != 1 {
			return domain.GenerateRequest{}, false
		}
		s, ok := v[0].(string)
		if !ok {
			return domain.GenerateRequest{}, false
		}
		prompt = s
	default:
		return domain.GenerateRequest{}, false
	}

	return domain.GenerateRequest{
		Model:            apiReq.Model,
		Prompt:           prompt,
		Suffix:           apiReq.Suffix,
		Stream:           apiReq.Stream,
		Temperature:      apiReq.Temperature,
		TopP:             apiReq.TopP,
		MaxTokens:        apiReq.MaxTokens,
		Stop:             NormalizeStop(apiReq.Stop),
		Seed:             apiReq.Seed,
		PresencePenalty:  apiReq.PresencePenalty,
		FrequencyPenalty: apiReq.FrequencyPenalty,
	}, true
}

// DomainToOpenAICompletion 将领域 ChatResponse 转换为旧版补全响应。
func DomainToOpenAICompletion(model string, resp *domain.ChatResponse) OpenAICompletionResponse {
	finishReason := MapOpenAIFinishReason(resp.FinishReason, false)
	return OpenAICompletionResponse{
		ID:      fmt.Sprintf("cmpl-%d", IDTimestamp()),
		Object:  "text_completion",
		Created: NowUnix(),
		Model:   model,
		Choices: []OpenAICompletionChoice{
			{Text: resp.Content, Index: 0, FinishReason: &finishReason},
		},
		Usage: newOpenAIUsage(resp.InputTokens, resp.OutputTokens),
	}
}

// BuildOpenAICompletionChunk 构建旧版补全 SSE 块。
// 当块不包含任何可发送内容时返回 false。
func BuildOpenAICompletionChunk(id string, created int64, model string, chunk domain.StreamChunk) (OpenAICompletionResponse, bool) {
	base := OpenAICompletionResponse{
		ID:      id,
		Object:  "text_completion",
		Created: created,
		Model:   model,
	}

	if chunk.Done {
		finishReason := "stop"
		if chunk.FinishReason != nil {
			finishReason = *chunk.FinishReason
		}
		finishReason = MapOpenAIFinishReason(finishReason, false)
		base.Choices = []OpenAICompletionChoice{
			{Text: "", Index: 0, FinishReason: &finishReason},
		}
		return base, true
	}

	if chunk.Content == "" {
		return OpenAICompletionResponse{}, false
	}
	base.Choices = []OpenAICompletionChoice{
		{Text: chunk.Content, Index: 0, FinishReason: nil},
	}
	return base, true
}

// MapOpenAIFinishReason 将 Ollama done_reason 映射为 OpenAI finish_reason。
func MapOpenAIFinishReason(reason string, hasToolCalls bool) string {
	if hasToolCalls {
		return "tool_calls"
	}
	switch reason {
	case "length", "max_tokens":
		return "length"
	default:
		return "stop"
	}
}

// ========== 辅助函数 ==========

func toOpenAIToolCalls(calls []domain.ToolCall, streaming bool) []OpenAIToolCall {
	if len(calls) == 0 {
		return nil
	}
	result := make([]OpenAIToolCall, len(calls))
	for i, tc := range calls {
		result[i] = OpenAIToolCall{
			ID:   tc.ID,
			Type: "function",
			Function: OpenAIToolCallFunction{
				Name:      tc.Name,
				Arguments: tc.Arguments,
			},
		}
	}
	return result
}

func newOpenAIUsage(prompt, completion int) *OpenAIUsage {
	if prompt == 0 && completion == 0 {
		return nil
	}
	return &OpenAIUsage{
		PromptTokens:     prompt,
		CompletionTokens: completion,
		TotalTokens:      prompt + completion,
	}
}

// NormalizeOpenAIContent 将 OpenAI 内容（字符串或部分数组）
// 转换为纯文本以及从 image_url 部分提取的 base64 图片。
func NormalizeOpenAIContent(content interface{}) (string, []string) {
	switch v := content.(type) {
	case string:
		return v, nil
	case []interface{}:
		var textParts []string
		var images []string
		for _, part := range v {
			m, ok := part.(map[string]interface{})
			if !ok {
				continue
			}
			switch m["type"] {
			case "text":
				if txt, ok := m["text"].(string); ok {
					textParts = append(textParts, txt)
				}
			case "image_url":
				if iu, ok := m["image_url"].(map[string]interface{}); ok {
					if url, ok := iu["url"].(string); ok {
						if img := ExtractBase64Image(url); img != "" {
							images = append(images, img)
						} else if IsRemoteURL(url) {
							// 保留为 URL；处理器会在请求到达 Ollama 之前下载它。
							images = append(images, url)
						}
					}
				}
			}
		}
		return strings.Join(textParts, ""), images
	case nil:
		return "", nil
	default:
		return fmt.Sprintf("%v", content), nil
	}
}

// NormalizeContent 将内容（字符串或部分数组）转换为纯字符串。
func NormalizeContent(content interface{}) string {
	text, _ := NormalizeOpenAIContent(content)
	return text
}
