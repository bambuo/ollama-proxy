package main

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
	"time"
)

// ---------- Request Types ----------

type StreamOptions struct {
	IncludeUsage bool `json:"include_usage"`
}

type ContentPart struct {
	Type     string `json:"type"`
	Text     string `json:"text,omitempty"`
	ImageURL *struct {
		URL string `json:"url"`
	} `json:"image_url,omitempty"`
}

type ChatMessage struct {
	Role    string      `json:"role"`
	Content interface{} `json:"content"`
}

// normalizeContent converts content (string or array of parts) to a plain string
func normalizeContent(content interface{}) string {
	switch v := content.(type) {
	case string:
		return v
	case []interface{}:
		var textParts []string
		for _, part := range v {
			if m, ok := part.(map[string]interface{}); ok {
				if m["type"] == "text" {
					if txt, ok := m["text"].(string); ok {
						textParts = append(textParts, txt)
					}
				}
			}
		}
		return strings.Join(textParts, "")
	default:
		return fmt.Sprintf("%v", content)
	}
}

type ChatRequest struct {
	Model          string         `json:"model"`
	Messages       []ChatMessage  `json:"messages"`
	Stream         bool           `json:"stream"`
	StreamOptions  *StreamOptions `json:"stream_options,omitempty"`
	Temperature    *float64       `json:"temperature,omitempty"`
	TopP           *float64       `json:"top_p,omitempty"`
	MaxTokens      *int           `json:"max_tokens,omitempty"`
	FrequencyPenal *float64       `json:"frequency_penalty,omitempty"`
	PresencePenal  *float64       `json:"presence_penalty,omitempty"`
	Seed           *int           `json:"seed,omitempty"`
}

// ---------- Ollama API Types ----------

type OllamaOptions struct {
	Temperature    float64 `json:"temperature,omitempty"`
	TopP           float64 `json:"top_p,omitempty"`
	NumPredict     int     `json:"num_predict,omitempty"`
	FrequencyPenal float64 `json:"frequency_penalty,omitempty"`
	PresencePenal  float64 `json:"presence_penalty,omitempty"`
	Seed           int     `json:"seed,omitempty"`
}

type OllamaChatRequest struct {
	Model     string         `json:"model"`
	Messages  []ChatMessage  `json:"messages"`
	Stream    bool           `json:"stream"`
	Options   *OllamaOptions `json:"options,omitempty"`
	KeepAlive string         `json:"keep_alive,omitempty"`
}

type OllamaChatResponse struct {
	Model     string `json:"model"`
	CreatedAt string `json:"created_at"`
	Message   *struct {
		Role    string `json:"role"`
		Content string `json:"content"`
	} `json:"message,omitempty"`
	DoneReason      string `json:"done_reason"`
	Done            bool   `json:"done"`
	EvalCount       int    `json:"eval_count"`
	PromptEvalCount int    `json:"prompt_eval_count"`
	TotalDuration   int64  `json:"total_duration"`
}

// ---------- OpenAI Response Types ----------

type OpenAIStreamDelta struct {
	Role    string `json:"role,omitempty"`
	Content string `json:"content,omitempty"`
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

type OpenAIChoice struct {
	Index        int               `json:"index"`
	Message      OpenAIStreamDelta `json:"message"`
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

// ---------- OpenAI Error Types ----------

type OpenAIError struct {
	Message string  `json:"message"`
	Type    string  `json:"type"`
	Param   *string `json:"param"`
	Code    *string `json:"code"`
}

type OpenAIErrorResponse struct {
	Error OpenAIError `json:"error"`
}

const systemFingerprint = "fp_ollama"

// ---------- Main Handler ----------

func handler(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		w.Header().Set("Allow", "POST")
		writeError(w, http.StatusMethodNotAllowed, "Only POST method is allowed")
		return
	}

	body, err := io.ReadAll(r.Body)
	if err != nil {
		writeError(w, http.StatusBadRequest, err.Error())
		return
	}

	var chatReq ChatRequest
	if err := json.Unmarshal(body, &chatReq); err != nil {
		writeError(w, http.StatusBadRequest, err.Error())
		return
	}

	if chatReq.Model == "" {
		writeError(w, http.StatusBadRequest, "model is required")
		return
	}

	if chatReq.Stream {
		handleStream(w, r, chatReq)
	} else {
		handleNonStream(w, r, chatReq)
	}
}

// ---------- Helpers: Build Ollama Request ----------

func buildOllamaReq(chatReq ChatRequest) OllamaChatRequest {
	// Normalize message content before sending to Ollama
	normalizedMessages := make([]ChatMessage, len(chatReq.Messages))
	for i, msg := range chatReq.Messages {
		normalizedMessages[i] = ChatMessage{
			Role:    msg.Role,
			Content: normalizeContent(msg.Content),
		}
	}

	ollamaReq := OllamaChatRequest{
		Model:     chatReq.Model,
		Messages:  normalizedMessages,
		Stream:    chatReq.Stream,
		KeepAlive: "5m",
	}

	// Pass through common parameters as Ollama options
	if chatReq.Temperature != nil || chatReq.TopP != nil || chatReq.MaxTokens != nil ||
		chatReq.FrequencyPenal != nil || chatReq.PresencePenal != nil || chatReq.Seed != nil {
		opts := &OllamaOptions{}
		if chatReq.Temperature != nil {
			opts.Temperature = *chatReq.Temperature
		}
		if chatReq.TopP != nil {
			opts.TopP = *chatReq.TopP
		}
		if chatReq.MaxTokens != nil {
			opts.NumPredict = *chatReq.MaxTokens
		}
		if chatReq.FrequencyPenal != nil {
			opts.FrequencyPenal = *chatReq.FrequencyPenal
		}
		if chatReq.PresencePenal != nil {
			opts.PresencePenal = *chatReq.PresencePenal
		}
		if chatReq.Seed != nil {
			opts.Seed = *chatReq.Seed
		}
		ollamaReq.Options = opts
	}

	return ollamaReq
}

func newUsage(prompt, completion int) *OpenAIUsage {
	if prompt == 0 && completion == 0 {
		return nil
	}
	return &OpenAIUsage{
		PromptTokens:     prompt,
		CompletionTokens: completion,
		TotalTokens:      prompt + completion,
	}
}

// ---------- Streaming ----------

func handleStream(w http.ResponseWriter, r *http.Request, chatReq ChatRequest) {
	ollamaReq := buildOllamaReq(chatReq)
	ollamaReq.Stream = true

	resp, err := callOllama(r.Context(), ollamaReq)
	if err != nil {
		writeError(w, http.StatusInternalServerError, err.Error())
		return
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		bodyBytes, _ := io.ReadAll(resp.Body)
		writeError(w, http.StatusBadGateway, fmt.Sprintf("Ollama returned status %d: %s", resp.StatusCode, string(bodyBytes)))
		return
	}

	// SSE headers
	w.Header().Set("Content-Type", "text/event-stream; charset=utf-8")
	w.Header().Set("Cache-Control", "no-cache")
	w.Header().Set("Connection", "keep-alive")
	w.Header().Set("X-Accel-Buffering", "no")

	flusher, ok := w.(http.Flusher)
	if !ok {
		writeError(w, http.StatusInternalServerError, "streaming not supported")
		return
	}

	includeUsage := chatReq.StreamOptions != nil && chatReq.StreamOptions.IncludeUsage
	created := time.Now().Unix()
	id := fmt.Sprintf("chatcmpl-%d", time.Now().UnixNano())

	scanner := bufio.NewScanner(resp.Body)
	buf := make([]byte, 0, 64*1024)
	scanner.Buffer(buf, 1024*1024)

	sentRole := false
	readDone := make(chan struct{}, 1)

	go func() {
		defer func() { readDone <- struct{}{} }()

		for scanner.Scan() {
			line := scanner.Bytes()
			if len(line) == 0 {
				continue
			}

			var ollamaResp OllamaChatResponse
			if err := json.Unmarshal(line, &ollamaResp); err != nil {
				continue
			}

			// Final chunk — includes finish_reason and metadata
			if ollamaResp.Done {
				finishReason := ollamaResp.DoneReason
				if finishReason == "" {
					finishReason = "stop"
				}

				chunk := OpenAIChunk{
					ID:                id,
					Object:            "chat.completion.chunk",
					Created:           created,
					Model:             chatReq.Model,
					SystemFingerprint: systemFingerprint,
					Choices: []OpenAIStreamChoice{
						{
							Index:        0,
							Delta:        OpenAIStreamDelta{},
							FinishReason: &finishReason,
						},
					},
				}
				writeSSE(w, flusher, chunk)

				// If stream_options: include_usage, send an additional chunk with usage
				if includeUsage {
					usageChunk := OpenAIChunk{
						ID:                id,
						Object:            "chat.completion.chunk",
						Created:           created,
						Model:             chatReq.Model,
						SystemFingerprint: systemFingerprint,
						Choices:           []OpenAIStreamChoice{},
						Usage:             newUsage(ollamaResp.PromptEvalCount, ollamaResp.EvalCount),
					}
					writeSSE(w, flusher, usageChunk)
				}

				return
			}

			if ollamaResp.Message == nil {
				continue
			}

			delta := OpenAIStreamDelta{}
			if !sentRole {
				delta.Role = "assistant"
				sentRole = true
			}
			delta.Content = ollamaResp.Message.Content

			// Skip empty content from role-only messages
			if delta.Content == "" && sentRole {
				continue
			}

			chunk := OpenAIChunk{
				ID:                id,
				Object:            "chat.completion.chunk",
				Created:           created,
				Model:             chatReq.Model,
				SystemFingerprint: systemFingerprint,
				Choices: []OpenAIStreamChoice{
					{
						Index:        0,
						Delta:        delta,
						FinishReason: nil,
					},
				},
			}
			writeSSE(w, flusher, chunk)
		}
	}()

	select {
	case <-readDone:
		// Normal completion
	case <-r.Context().Done():
		// Client disconnected
		return
	}

	// Signal end of stream
	fmt.Fprintf(w, "data: [DONE]\n\n")
	flusher.Flush()
}

// ---------- Non-Streaming ----------

func handleNonStream(w http.ResponseWriter, r *http.Request, chatReq ChatRequest) {
	ollamaReq := buildOllamaReq(chatReq)
	ollamaReq.Stream = false

	resp, err := callOllama(r.Context(), ollamaReq)
	if err != nil {
		writeError(w, http.StatusInternalServerError, err.Error())
		return
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		bodyBytes, _ := io.ReadAll(resp.Body)
		writeError(w, http.StatusBadGateway, fmt.Sprintf("Ollama returned status %d: %s", resp.StatusCode, string(bodyBytes)))
		return
	}

	respBody, err := io.ReadAll(resp.Body)
	if err != nil {
		writeError(w, http.StatusInternalServerError, err.Error())
		return
	}

	var ollamaResp OllamaChatResponse
	if err := json.Unmarshal(respBody, &ollamaResp); err != nil {
		writeError(w, http.StatusInternalServerError, err.Error())
		return
	}

	if ollamaResp.Message == nil {
		writeError(w, http.StatusInternalServerError, "empty response from Ollama")
		return
	}

	finishReason := ollamaResp.DoneReason
	if finishReason == "" {
		finishReason = "stop"
	}

	openaiResp := OpenAIResponse{
		ID:                fmt.Sprintf("chatcmpl-%d", time.Now().UnixNano()),
		Object:            "chat.completion",
		Created:           time.Now().Unix(),
		Model:             chatReq.Model,
		SystemFingerprint: systemFingerprint,
		Choices: []OpenAIChoice{
			{
				Index: 0,
				Message: OpenAIStreamDelta{
					Role:    "assistant",
					Content: ollamaResp.Message.Content,
				},
				FinishReason: finishReason,
			},
		},
		Usage: newUsage(ollamaResp.PromptEvalCount, ollamaResp.EvalCount),
	}

	w.Header().Set("Content-Type", "application/json; charset=utf-8")
	json.NewEncoder(w).Encode(openaiResp)
}

// ---------- Helpers ----------

func callOllama(ctx context.Context, req OllamaChatRequest) (*http.Response, error) {
	reqBody, err := json.Marshal(req)
	if err != nil {
		return nil, fmt.Errorf("marshal request: %w", err)
	}

	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost,
		"http://127.0.0.1:11434/api/chat", bytes.NewBuffer(reqBody))
	if err != nil {
		return nil, fmt.Errorf("create request: %w", err)
	}
	httpReq.Header.Set("Content-Type", "application/json")

	client := &http.Client{
		Timeout: 0, // no timeout, rely on context
	}
	return client.Do(httpReq)
}

func writeSSE(w http.ResponseWriter, flusher http.Flusher, data interface{}) {
	encoded, err := json.Marshal(data)
	if err != nil {
		return
	}
	fmt.Fprintf(w, "data: %s\n\n", encoded)
	flusher.Flush()
}

func writeError(w http.ResponseWriter, status int, msg string) {
	errType := "server_error"
	switch {
	case status == http.StatusBadRequest:
		errType = "invalid_request_error"
	case status == http.StatusMethodNotAllowed:
		errType = "invalid_request_error"
	case status == http.StatusNotFound:
		errType = "not_found"
	case status >= 500:
		errType = "server_error"
	}

	errBody, _ := json.Marshal(OpenAIErrorResponse{
		Error: OpenAIError{
			Message: msg,
			Type:    errType,
			Param:   nil,
			Code:    nil,
		},
	})

	w.Header().Set("Content-Type", "application/json; charset=utf-8")
	w.WriteHeader(status)
	w.Write(errBody)
}

func debugHandler(w http.ResponseWriter, r *http.Request) {
	// Accept both /v1/chat/completions and /chat/completions
	if r.URL.Path == "/v1/chat/completions" || r.URL.Path == "/chat/completions" {
		handler(w, r)
		return
	}
	// Otherwise return 404 with the path info in response
	writeError(w, http.StatusNotFound, fmt.Sprintf("path not found: %s", r.URL.Path))
}

func main() {
	http.HandleFunc("/", debugHandler)
	fmt.Println("✅ Ollama Proxy running at http://127.0.0.1:3000")
	fmt.Println("   Listening for requests...")
	http.ListenAndServe(":3000", nil)
}
