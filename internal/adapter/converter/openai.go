package converter

import (
	"encoding/json"
	"fmt"
	"strings"

	"ollama-proxy/internal/domain"
)

// ========== OpenAI Request DTO ==========

type OpenAIToolCallFunction struct {
	Name      string `json:"name,omitempty"`
	Arguments string `json:"arguments,omitempty"`
}

type OpenAIToolCall struct {
	Index    *int                   `json:"index,omitempty"` // present in streaming deltas
	ID       string                 `json:"id,omitempty"`
	Type     string                 `json:"type,omitempty"`
	Function OpenAIToolCallFunction `json:"function"`
}

type OpenAIMessage struct {
	Role             string           `json:"role"`
	Content          interface{}      `json:"content"` // string or []ContentPart
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
	Stop                interface{}           `json:"stop,omitempty"` // string or []string
	Seed                *int                  `json:"seed,omitempty"`
	PresencePenalty     *float64              `json:"presence_penalty,omitempty"`
	FrequencyPenalty    *float64              `json:"frequency_penalty,omitempty"`
	Tools               []OpenAITool          `json:"tools,omitempty"`
	ToolChoice          interface{}           `json:"tool_choice,omitempty"` // "none"/"auto"/"required" or object
	ResponseFormat      *OpenAIResponseFormat `json:"response_format,omitempty"`
}

// ========== OpenAI Response DTO ==========

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

// ========== OpenAI Models DTO ==========

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

// ========== OpenAI Legacy Completions DTO ==========

type OpenAICompletionRequest struct {
	Model            string               `json:"model"`
	Prompt           interface{}          `json:"prompt"` // string or []string
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

// ========== OpenAI Embeddings DTO ==========

type OpenAIEmbeddingRequest struct {
	Model string      `json:"model"`
	Input interface{} `json:"input"` // string or []string
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

// ========== Conversion Functions ==========

// OpenAIRequestToDomain converts an OpenAI request to a domain ChatRequest.
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

	// tool_choice "none" disables tools; Ollama cannot force a specific
	// tool, so other choices pass tools through as-is.
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

// DomainToOpenAIResponse converts a domain ChatResponse to an OpenAI response.
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

// BuildOpenAIStreamChunk constructs an OpenAI SSE chunk from a domain
// StreamChunk. toolCallIndex tracks tool-call indices across the stream.
// Returns false when the chunk carries nothing to send.
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

// BuildOpenAIUsageChunk creates a usage-only SSE chunk.
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

// DomainToOpenAIModelList converts model info to the OpenAI models format.
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

// DomainToOpenAIEmbeddings converts an embedding result to OpenAI format.
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

// ResponseFormatToOllama maps an OpenAI response_format to the Ollama format
// parameter: "json" for json_object, the raw schema for json_schema.
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

// CompletionRequestToDomain converts a legacy completions request to a domain
// GenerateRequest. Returns ok=false when prompt is not a single string.
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

// DomainToOpenAICompletion converts a domain ChatResponse to a legacy
// completions response.
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

// BuildOpenAICompletionChunk constructs a legacy completions SSE chunk.
// Returns false when the chunk carries nothing to send.
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

// MapOpenAIFinishReason maps an Ollama done_reason to an OpenAI finish_reason.
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

// ========== Helpers ==========

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

// NormalizeOpenAIContent converts OpenAI content (string or array of parts)
// to plain text plus any base64 images found in image_url parts.
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
							// Kept as a URL; the handler downloads it
							// before the request reaches Ollama.
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

// NormalizeContent converts content (string or array of parts) to a plain string.
func NormalizeContent(content interface{}) string {
	text, _ := NormalizeOpenAIContent(content)
	return text
}
