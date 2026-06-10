package domain

import "encoding/json"

// ValidationError marks a request the target model cannot serve (e.g.
// images sent to a non-vision model). Handlers map it to HTTP 400.
type ValidationError string

func (e ValidationError) Error() string { return string(e) }

// ToolCall represents a tool/function invocation requested by the model.
type ToolCall struct {
	ID        string
	Name      string
	Arguments string // JSON-encoded arguments object
}

// Tool describes a function the model is allowed to call.
type Tool struct {
	Name        string
	Description string
	Parameters  json.RawMessage // JSON schema of the arguments
}

// Message represents a single message in a chat conversation,
// protocol-agnostic with pre-normalized string content.
type Message struct {
	Role       string
	Content    string
	Thinking   string     // reasoning content on assistant messages
	Images     []string   // base64-encoded images, no data-URL prefix
	ToolCalls  []ToolCall // set on assistant messages that requested tools
	ToolCallID string     // set on tool-result messages (role "tool")
	ToolName   string     // optional tool name on tool-result messages
}

// ChatRequest represents a protocol-agnostic chat completion request.
type ChatRequest struct {
	Model            string
	Messages         []Message
	Stream           bool
	Temperature      *float64
	TopP             *float64
	TopK             *int
	MaxTokens        *int
	Stop             []string
	Seed             *int
	PresencePenalty  *float64
	FrequencyPenalty *float64
	Tools            []Tool
	Think            *bool           // enable/disable reasoning on thinking models
	Format           json.RawMessage // `"json"` or a JSON schema for structured output
}

// GenerateRequest represents a raw text completion request (no chat template
// beyond the model's own prompt template).
type GenerateRequest struct {
	Model            string
	Prompt           string
	Suffix           string
	Stream           bool
	Temperature      *float64
	TopP             *float64
	MaxTokens        *int
	Stop             []string
	Seed             *int
	PresencePenalty  *float64
	FrequencyPenalty *float64
	Format           json.RawMessage
}

const (
	charsPerToken  = 4   // rough average for estimation
	tokensPerImage = 768 // ballpark vision token cost per image
)

// EstimateInputTokens roughly estimates the prompt token count of the
// request (~4 chars per token, flat cost per image). Used for sizing the
// backend context window; intentionally errs on the simple side.
func (r *ChatRequest) EstimateInputTokens() int {
	chars := 0
	images := 0
	for _, m := range r.Messages {
		chars += len(m.Role) + len(m.Content) + len(m.Thinking)
		for _, tc := range m.ToolCalls {
			chars += len(tc.Name) + len(tc.Arguments)
		}
		images += len(m.Images)
	}
	for _, t := range r.Tools {
		chars += len(t.Name) + len(t.Description) + len(t.Parameters)
	}
	return chars/charsPerToken + images*tokensPerImage
}

// EstimateInputTokens roughly estimates the prompt token count.
func (r *GenerateRequest) EstimateInputTokens() int {
	return (len(r.Prompt) + len(r.Suffix)) / charsPerToken
}

// ChatResponse represents a protocol-agnostic non-streaming response.
type ChatResponse struct {
	Content      string
	Thinking     string
	ToolCalls    []ToolCall
	FinishReason string
	InputTokens  int
	OutputTokens int
	Model        string
}

// StreamChunk represents a single chunk in a streaming response.
// Content carries the incremental text delta of this chunk.
type StreamChunk struct {
	Content      string
	Thinking     string
	ToolCalls    []ToolCall
	FinishReason *string
	Done         bool
	InputTokens  int
	OutputTokens int
}

// ModelInfo describes a model available on the backend.
type ModelInfo struct {
	Name       string
	ModifiedAt string
	Size       int64
}

// EmbeddingResult holds embeddings for a batch of inputs.
type EmbeddingResult struct {
	Embeddings  [][]float64
	InputTokens int
	Model       string
}
