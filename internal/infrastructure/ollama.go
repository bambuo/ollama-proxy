package infrastructure

import (
	"bufio"
	"bytes"
	"context"
	"crypto/rand"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"

	"ollama-proxy/internal/application"
	"ollama-proxy/internal/domain"
)

// DefaultOllamaURL is the default address of the local Ollama API.
const DefaultOllamaURL = "http://127.0.0.1:11434"

// Ollama internal types (not exported).

type ollamaToolCallFunction struct {
	Name      string          `json:"name"`
	Arguments json.RawMessage `json:"arguments,omitempty"`
}

type ollamaToolCall struct {
	Function ollamaToolCallFunction `json:"function"`
}

type ollamaMessage struct {
	Role      string           `json:"role"`
	Content   string           `json:"content"`
	Thinking  string           `json:"thinking,omitempty"`
	Images    []string         `json:"images,omitempty"`
	ToolCalls []ollamaToolCall `json:"tool_calls,omitempty"`
}

type ollamaToolFunction struct {
	Name        string          `json:"name"`
	Description string          `json:"description,omitempty"`
	Parameters  json.RawMessage `json:"parameters,omitempty"`
}

type ollamaTool struct {
	Type     string             `json:"type"`
	Function ollamaToolFunction `json:"function"`
}

type ollamaOptions struct {
	Temperature      *float64 `json:"temperature,omitempty"`
	TopP             *float64 `json:"top_p,omitempty"`
	TopK             *int     `json:"top_k,omitempty"`
	NumPredict       *int     `json:"num_predict,omitempty"`
	Stop             []string `json:"stop,omitempty"`
	Seed             *int     `json:"seed,omitempty"`
	PresencePenalty  *float64 `json:"presence_penalty,omitempty"`
	FrequencyPenalty *float64 `json:"frequency_penalty,omitempty"`
}

type ollamaRequest struct {
	Model     string          `json:"model"`
	Messages  []ollamaMessage `json:"messages"`
	Stream    bool            `json:"stream"`
	Tools     []ollamaTool    `json:"tools,omitempty"`
	Think     *bool           `json:"think,omitempty"`
	Format    json.RawMessage `json:"format,omitempty"`
	KeepAlive string          `json:"keep_alive,omitempty"`
	Options   *ollamaOptions  `json:"options,omitempty"`
}

type ollamaGenerateRequest struct {
	Model     string          `json:"model"`
	Prompt    string          `json:"prompt"`
	Suffix    string          `json:"suffix,omitempty"`
	Stream    bool            `json:"stream"`
	Format    json.RawMessage `json:"format,omitempty"`
	KeepAlive string          `json:"keep_alive,omitempty"`
	Options   *ollamaOptions  `json:"options,omitempty"`
}

type ollamaResponse struct {
	Model           string         `json:"model"`
	CreatedAt       string         `json:"created_at"`
	Message         *ollamaMessage `json:"message,omitempty"`
	Response        string         `json:"response,omitempty"` // /api/generate
	Thinking        string         `json:"thinking,omitempty"` // /api/generate
	DoneReason      string         `json:"done_reason"`
	Done            bool           `json:"done"`
	EvalCount       int            `json:"eval_count"`
	PromptEvalCount int            `json:"prompt_eval_count"`
}

type ollamaTagsResponse struct {
	Models []struct {
		Name       string `json:"name"`
		ModifiedAt string `json:"modified_at"`
		Size       int64  `json:"size"`
	} `json:"models"`
}

type ollamaEmbedRequest struct {
	Model string   `json:"model"`
	Input []string `json:"input"`
}

type ollamaEmbedResponse struct {
	Model           string      `json:"model"`
	Embeddings      [][]float64 `json:"embeddings"`
	PromptEvalCount int         `json:"prompt_eval_count"`
}

type ollamaClient struct {
	baseURL string
	client  *http.Client
}

// NewOllamaClient creates an OllamaClient that implements application.OllamaClient.
func NewOllamaClient() application.OllamaClient {
	url := os.Getenv("OLLAMA_URL")
	if url == "" {
		url = DefaultOllamaURL
	}
	return &ollamaClient{
		baseURL: url,
		client: &http.Client{
			Timeout: 0, // rely on context
		},
	}
}

func (c *ollamaClient) Chat(ctx context.Context, req *domain.ChatRequest) (*domain.ChatResponse, error) {
	respBody, err := c.post(ctx, "/api/chat", c.buildRequest(req, false))
	if err != nil {
		return nil, err
	}

	var ollamaResp ollamaResponse
	if err := json.Unmarshal(respBody, &ollamaResp); err != nil {
		return nil, fmt.Errorf("parse response: %w", err)
	}

	content := ""
	thinking := ""
	var toolCalls []domain.ToolCall
	if ollamaResp.Message != nil {
		content = ollamaResp.Message.Content
		thinking = ollamaResp.Message.Thinking
		toolCalls = toDomainToolCalls(ollamaResp.Message.ToolCalls)
	}

	finishReason := ollamaResp.DoneReason
	if finishReason == "" {
		finishReason = "stop"
	}

	return &domain.ChatResponse{
		Content:      content,
		Thinking:     thinking,
		ToolCalls:    toolCalls,
		FinishReason: finishReason,
		InputTokens:  ollamaResp.PromptEvalCount,
		OutputTokens: ollamaResp.EvalCount,
		Model:        ollamaResp.Model,
	}, nil
}

func (c *ollamaClient) ChatStream(ctx context.Context, req *domain.ChatRequest) (<-chan domain.StreamChunk, error) {
	return c.stream(ctx, "/api/chat", c.buildRequest(req, true))
}

// Generate performs a raw completion via /api/generate.
func (c *ollamaClient) Generate(ctx context.Context, req *domain.GenerateRequest) (*domain.ChatResponse, error) {
	respBody, err := c.post(ctx, "/api/generate", c.buildGenerateRequest(req, false))
	if err != nil {
		return nil, err
	}

	var ollamaResp ollamaResponse
	if err := json.Unmarshal(respBody, &ollamaResp); err != nil {
		return nil, fmt.Errorf("parse response: %w", err)
	}

	finishReason := ollamaResp.DoneReason
	if finishReason == "" {
		finishReason = "stop"
	}

	return &domain.ChatResponse{
		Content:      ollamaResp.Response,
		Thinking:     ollamaResp.Thinking,
		FinishReason: finishReason,
		InputTokens:  ollamaResp.PromptEvalCount,
		OutputTokens: ollamaResp.EvalCount,
		Model:        ollamaResp.Model,
	}, nil
}

// GenerateStream performs a streaming raw completion via /api/generate.
func (c *ollamaClient) GenerateStream(ctx context.Context, req *domain.GenerateRequest) (<-chan domain.StreamChunk, error) {
	return c.stream(ctx, "/api/generate", c.buildGenerateRequest(req, true))
}

// stream posts an NDJSON streaming request and converts each line into a
// domain StreamChunk. Works for both /api/chat and /api/generate.
func (c *ollamaClient) stream(ctx context.Context, path string, payload interface{}) (<-chan domain.StreamChunk, error) {
	body, err := json.Marshal(payload)
	if err != nil {
		return nil, fmt.Errorf("marshal ollama request: %w", err)
	}

	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost,
		c.baseURL+path, bytes.NewBuffer(body))
	if err != nil {
		return nil, fmt.Errorf("create stream request: %w", err)
	}
	httpReq.Header.Set("Content-Type", "application/json")

	resp, err := c.client.Do(httpReq)
	if err != nil {
		return nil, fmt.Errorf("ollama stream request: %w", err)
	}

	if resp.StatusCode != http.StatusOK {
		bodyBytes, _ := io.ReadAll(resp.Body)
		resp.Body.Close()
		return nil, fmt.Errorf("ollama stream status %d: %s", resp.StatusCode, string(bodyBytes))
	}

	chunkChan := make(chan domain.StreamChunk, 10)

	go func() {
		defer resp.Body.Close()
		defer close(chunkChan)

		scanner := bufio.NewScanner(resp.Body)
		buf := make([]byte, 0, 64*1024)
		scanner.Buffer(buf, 1024*1024)

		for scanner.Scan() {
			line := scanner.Bytes()
			if len(line) == 0 {
				continue
			}

			var ollamaResp ollamaResponse
			if err := json.Unmarshal(line, &ollamaResp); err != nil {
				continue
			}

			chunk := domain.StreamChunk{
				Done:         ollamaResp.Done,
				InputTokens:  ollamaResp.PromptEvalCount,
				OutputTokens: ollamaResp.EvalCount,
			}

			if ollamaResp.Done {
				reason := ollamaResp.DoneReason
				if reason == "" {
					reason = "stop"
				}
				chunk.FinishReason = &reason
			}

			chunk.Content = ollamaResp.Response
			chunk.Thinking = ollamaResp.Thinking
			if ollamaResp.Message != nil {
				chunk.Content = ollamaResp.Message.Content
				chunk.Thinking = ollamaResp.Message.Thinking
				chunk.ToolCalls = toDomainToolCalls(ollamaResp.Message.ToolCalls)
			}

			select {
			case chunkChan <- chunk:
			case <-ctx.Done():
				return
			}
		}
	}()

	return chunkChan, nil
}

func (c *ollamaClient) ListModels(ctx context.Context) ([]domain.ModelInfo, error) {
	httpReq, err := http.NewRequestWithContext(ctx, http.MethodGet, c.baseURL+"/api/tags", nil)
	if err != nil {
		return nil, fmt.Errorf("create request: %w", err)
	}

	resp, err := c.client.Do(httpReq)
	if err != nil {
		return nil, fmt.Errorf("ollama request: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		bodyBytes, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("ollama status %d: %s", resp.StatusCode, string(bodyBytes))
	}

	var tags ollamaTagsResponse
	if err := json.NewDecoder(resp.Body).Decode(&tags); err != nil {
		return nil, fmt.Errorf("parse response: %w", err)
	}

	models := make([]domain.ModelInfo, len(tags.Models))
	for i, m := range tags.Models {
		models[i] = domain.ModelInfo{
			Name:       m.Name,
			ModifiedAt: m.ModifiedAt,
			Size:       m.Size,
		}
	}
	return models, nil
}

func (c *ollamaClient) Embed(ctx context.Context, model string, input []string) (*domain.EmbeddingResult, error) {
	respBody, err := c.post(ctx, "/api/embed", ollamaEmbedRequest{Model: model, Input: input})
	if err != nil {
		return nil, err
	}

	var embedResp ollamaEmbedResponse
	if err := json.Unmarshal(respBody, &embedResp); err != nil {
		return nil, fmt.Errorf("parse response: %w", err)
	}

	return &domain.EmbeddingResult{
		Embeddings:  embedResp.Embeddings,
		InputTokens: embedResp.PromptEvalCount,
		Model:       embedResp.Model,
	}, nil
}

func (c *ollamaClient) post(ctx context.Context, path string, payload interface{}) ([]byte, error) {
	body, err := json.Marshal(payload)
	if err != nil {
		return nil, fmt.Errorf("marshal ollama request: %w", err)
	}

	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost,
		c.baseURL+path, bytes.NewBuffer(body))
	if err != nil {
		return nil, fmt.Errorf("create request: %w", err)
	}
	httpReq.Header.Set("Content-Type", "application/json")

	resp, err := c.client.Do(httpReq)
	if err != nil {
		return nil, fmt.Errorf("ollama request: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		bodyBytes, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("ollama status %d: %s", resp.StatusCode, string(bodyBytes))
	}

	respBody, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("read response: %w", err)
	}
	return respBody, nil
}

func (c *ollamaClient) buildRequest(req *domain.ChatRequest, stream bool) ollamaRequest {
	messages := make([]ollamaMessage, len(req.Messages))
	for i, msg := range req.Messages {
		om := ollamaMessage{
			Role:     msg.Role,
			Content:  msg.Content,
			Thinking: msg.Thinking,
			Images:   msg.Images,
		}
		for _, tc := range msg.ToolCalls {
			args := json.RawMessage(tc.Arguments)
			if !json.Valid(args) {
				args = json.RawMessage("{}")
			}
			om.ToolCalls = append(om.ToolCalls, ollamaToolCall{
				Function: ollamaToolCallFunction{
					Name:      tc.Name,
					Arguments: args,
				},
			})
		}
		messages[i] = om
	}

	ollamaReq := ollamaRequest{
		Model:     req.Model,
		Messages:  messages,
		Stream:    stream,
		Think:     req.Think,
		Format:    req.Format,
		KeepAlive: "5m",
	}

	for _, t := range req.Tools {
		ollamaReq.Tools = append(ollamaReq.Tools, ollamaTool{
			Type: "function",
			Function: ollamaToolFunction{
				Name:        t.Name,
				Description: t.Description,
				Parameters:  t.Parameters,
			},
		})
	}

	if req.Temperature != nil || req.TopP != nil || req.TopK != nil || req.MaxTokens != nil ||
		len(req.Stop) > 0 || req.Seed != nil || req.PresencePenalty != nil || req.FrequencyPenalty != nil {
		ollamaReq.Options = &ollamaOptions{
			Temperature:      req.Temperature,
			TopP:             req.TopP,
			TopK:             req.TopK,
			NumPredict:       req.MaxTokens,
			Stop:             req.Stop,
			Seed:             req.Seed,
			PresencePenalty:  req.PresencePenalty,
			FrequencyPenalty: req.FrequencyPenalty,
		}
	}

	return ollamaReq
}

func (c *ollamaClient) buildGenerateRequest(req *domain.GenerateRequest, stream bool) ollamaGenerateRequest {
	genReq := ollamaGenerateRequest{
		Model:     req.Model,
		Prompt:    req.Prompt,
		Suffix:    req.Suffix,
		Stream:    stream,
		Format:    req.Format,
		KeepAlive: "5m",
	}

	if req.Temperature != nil || req.TopP != nil || req.MaxTokens != nil ||
		len(req.Stop) > 0 || req.Seed != nil || req.PresencePenalty != nil || req.FrequencyPenalty != nil {
		genReq.Options = &ollamaOptions{
			Temperature:      req.Temperature,
			TopP:             req.TopP,
			NumPredict:       req.MaxTokens,
			Stop:             req.Stop,
			Seed:             req.Seed,
			PresencePenalty:  req.PresencePenalty,
			FrequencyPenalty: req.FrequencyPenalty,
		}
	}

	return genReq
}

// toDomainToolCalls converts Ollama tool calls to domain tool calls,
// generating IDs since Ollama does not provide them.
func toDomainToolCalls(calls []ollamaToolCall) []domain.ToolCall {
	if len(calls) == 0 {
		return nil
	}
	result := make([]domain.ToolCall, len(calls))
	for i, tc := range calls {
		args := "{}"
		if len(tc.Function.Arguments) > 0 {
			args = string(tc.Function.Arguments)
		}
		result[i] = domain.ToolCall{
			ID:        newToolCallID(),
			Name:      tc.Function.Name,
			Arguments: args,
		}
	}
	return result
}

func newToolCallID() string {
	b := make([]byte, 12)
	if _, err := rand.Read(b); err != nil {
		return "call_fallback"
	}
	return "call_" + hex.EncodeToString(b)
}
