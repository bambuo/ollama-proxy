package domain

import "encoding/json"

// ValidationError 标记目标模型无法服务的请求（例如向非视觉模型发送图片）。
// 处理器将其映射为 HTTP 400。
type ValidationError string

func (e ValidationError) Error() string { return string(e) }

// ToolCall 表示模型请求的工具/函数调用。
type ToolCall struct {
	ID        string
	Name      string
	Arguments string // JSON 编码的参数对象
}

// Tool 描述模型允许调用的函数。
type Tool struct {
	Name        string
	Description string
	Parameters  json.RawMessage // 参数的 JSON Schema
}

// Message 表示聊天对话中的单条消息，协议无关且内容已预归一化为字符串。
type Message struct {
	Role       string
	Content    string
	Thinking   string     // 助手消息上的推理内容
	Images     []string   // base64 编码的图片，不含 data-URL 前缀
	ToolCalls  []ToolCall // 在请求工具的助手消息上设置
	ToolCallID string     // 在工具结果消息上设置（role "tool"）
	ToolName   string     // 工具结果消息上的可选工具名称
}

// ChatRequest 表示协议无关的聊天补全请求。
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
	Think            *bool           // 在推理模型上启用/禁用推理
	Format           json.RawMessage // `"json"` 或用于结构化输出的 JSON Schema
}

// GenerateRequest 表示原始文本补全请求（除了模型自身的提示模板外没有聊天模板）。
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
	charsPerToken  = 4   // 估算的粗略平均值
	tokensPerImage = 768 // 每张图片的大致视觉 token 消耗
)

// EstimateInputTokens 粗略估算请求的提示 token 数量（约 4 字符/token，每张图片固定成本）。
// 用于调整后端上下文窗口大小；有意偏向简单的估算。
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

// EstimateInputTokens 粗略估算提示 token 数量。
func (r *GenerateRequest) EstimateInputTokens() int {
	return (len(r.Prompt) + len(r.Suffix)) / charsPerToken
}

// ChatResponse 表示协议无关的非流式响应。
type ChatResponse struct {
	Content      string
	Thinking     string
	ToolCalls    []ToolCall
	FinishReason string
	InputTokens  int
	OutputTokens int
	Model        string
}

// StreamChunk 表示流式响应中的单个块。
// Content 携带此块的增量文本。
type StreamChunk struct {
	Content      string
	Thinking     string
	ToolCalls    []ToolCall
	FinishReason *string
	Done         bool
	InputTokens  int
	OutputTokens int
}

// ModelInfo 描述后端可用的模型。
type ModelInfo struct {
	Name       string
	ModifiedAt string
	Size       int64
}

// EmbeddingResult 保存一批输入的嵌入向量。
type EmbeddingResult struct {
	Embeddings  [][]float64
	InputTokens int
	Model       string
}
