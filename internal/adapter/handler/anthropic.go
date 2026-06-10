package handler

import (
	"encoding/json"
	"io"
	"net/http"

	"ollama-proxy/internal/adapter/converter"
	"ollama-proxy/internal/application"
	"ollama-proxy/internal/domain"
)

// AnthropicHandler handles Anthropic Messages API requests.
type AnthropicHandler struct {
	uc application.ChatUseCase
}

// NewAnthropicHandler creates a new AnthropicHandler.
func NewAnthropicHandler(uc application.ChatUseCase) *AnthropicHandler {
	return &AnthropicHandler{uc: uc}
}

// Handle processes Anthropic /v1/messages requests.
func (h *AnthropicHandler) Handle(w http.ResponseWriter, r *http.Request) {
	anthropicReq, ok := h.parseRequest(w, r)
	if !ok {
		return
	}

	if anthropicReq.MaxTokens <= 0 {
		WriteAnthropicError(w, http.StatusBadRequest, "max_tokens is required")
		return
	}

	domainReq := converter.AnthropicRequestToDomain(*anthropicReq)

	if err := resolveRemoteImages(r.Context(), &domainReq); err != nil {
		WriteAnthropicError(w, http.StatusBadRequest, err.Error())
		return
	}

	if domainReq.Stream {
		h.handleStream(w, r, &domainReq)
	} else {
		h.handleNonStream(w, r, &domainReq)
	}
}

// HandleCountTokens processes Anthropic /v1/messages/count_tokens requests.
func (h *AnthropicHandler) HandleCountTokens(w http.ResponseWriter, r *http.Request) {
	anthropicReq, ok := h.parseRequest(w, r)
	if !ok {
		return
	}

	w.Header().Set("Content-Type", "application/json; charset=utf-8")
	json.NewEncoder(w).Encode(converter.AnthropicCountTokensResponse{
		InputTokens: converter.EstimateAnthropicTokens(*anthropicReq),
	})
}

func (h *AnthropicHandler) parseRequest(w http.ResponseWriter, r *http.Request) (*converter.AnthropicRequest, bool) {
	if r.Method != http.MethodPost {
		w.Header().Set("Allow", "POST")
		WriteAnthropicError(w, http.StatusMethodNotAllowed, "Only POST method is allowed")
		return nil, false
	}

	body, err := io.ReadAll(http.MaxBytesReader(w, r.Body, maxRequestBodySize))
	if err != nil {
		WriteAnthropicError(w, http.StatusBadRequest, err.Error())
		return nil, false
	}

	var anthropicReq converter.AnthropicRequest
	if err := json.Unmarshal(body, &anthropicReq); err != nil {
		WriteAnthropicError(w, http.StatusBadRequest, err.Error())
		return nil, false
	}

	if anthropicReq.Model == "" {
		WriteAnthropicError(w, http.StatusBadRequest, "model is required")
		return nil, false
	}

	return &anthropicReq, true
}

func (h *AnthropicHandler) handleNonStream(w http.ResponseWriter, r *http.Request, domainReq *domain.ChatRequest) {
	resp, err := h.uc.Chat(r.Context(), domainReq)
	if err != nil {
		WriteAnthropicError(w, StatusForError(err), err.Error())
		return
	}

	anthropicResp := converter.DomainToAnthropicResponse(domainReq.Model, resp)

	w.Header().Set("Content-Type", "application/json; charset=utf-8")
	w.Header().Set("x-request-id", anthropicResp.ID)
	json.NewEncoder(w).Encode(anthropicResp)
}

func (h *AnthropicHandler) handleStream(w http.ResponseWriter, r *http.Request, domainReq *domain.ChatRequest) {
	// SSE headers
	w.Header().Set("Content-Type", "text/event-stream; charset=utf-8")
	w.Header().Set("Cache-Control", "no-cache")
	w.Header().Set("Connection", "keep-alive")
	w.Header().Set("X-Accel-Buffering", "no")

	flusher, ok := w.(http.Flusher)
	if !ok {
		WriteAnthropicError(w, http.StatusInternalServerError, "streaming not supported")
		return
	}

	msgID := "msg_" + converter.FormatHex()

	// message_start (followed by a ping, matching the real API's stream
	// shape) is deferred until the backend produces its first chunk, so
	// that pre-stream failures can still be reported as an HTTP error.
	started := false
	ensureStarted := func() {
		if started {
			return
		}
		started = true
		WriteAnthropicSSE(w, flusher, "message_start", converter.AnthropicMessageStart{
			Type: "message_start",
			Message: converter.AnthropicStartMsg{
				ID:           msgID,
				Type:         "message",
				Role:         "assistant",
				Content:      []converter.AnthropicContentBlock{},
				Model:        domainReq.Model,
				StopReason:   nil,
				StopSequence: nil,
				Usage:        converter.AnthropicUsage{},
			},
		})
		WriteAnthropicSSE(w, flusher, "ping", converter.AnthropicPing{Type: "ping"})
	}

	// Content blocks are opened lazily: a thinking block on the first
	// thinking delta, a text block on the first text delta, and one
	// tool_use block per tool call.
	blockIndex := 0
	openBlock := "" // "", "thinking" or "text"
	sawToolCalls := false

	closeBlock := func() {
		if openBlock != "" {
			WriteAnthropicSSE(w, flusher, "content_block_stop", converter.AnthropicContentBlockStop{
				Type:  "content_block_stop",
				Index: blockIndex,
			})
			blockIndex++
			openBlock = ""
		}
	}

	ensureBlock := func(kind string, start interface{}) {
		if openBlock == kind {
			return
		}
		closeBlock()
		WriteAnthropicSSE(w, flusher, "content_block_start", converter.AnthropicContentBlockStart{
			Type:         "content_block_start",
			Index:        blockIndex,
			ContentBlock: start,
		})
		openBlock = kind
	}

	err := h.uc.ChatStream(r.Context(), domainReq, func(chunk *domain.StreamChunk) error {
		select {
		case <-r.Context().Done():
			return r.Context().Err()
		default:
		}
		ensureStarted()

		if chunk.Done {
			closeBlock()

			stopReasonValue := "stop"
			if chunk.FinishReason != nil {
				stopReasonValue = *chunk.FinishReason
			}
			stopReason := converter.MapStopReason(stopReasonValue)
			if sawToolCalls {
				stopReason = "tool_use"
			}

			WriteAnthropicSSE(w, flusher, "message_delta", converter.AnthropicMessageDelta{
				Type: "message_delta",
				Delta: converter.AnthropicDelta{
					StopReason:   stopReason,
					StopSequence: nil,
				},
				Usage: &converter.AnthropicUsage{
					InputTokens:  chunk.InputTokens,
					OutputTokens: chunk.OutputTokens,
				},
			})

			WriteAnthropicSSE(w, flusher, "message_stop", converter.AnthropicMessageStop{
				Type: "message_stop",
			})
			return nil
		}

		if chunk.Thinking != "" {
			ensureBlock("thinking", converter.AnthropicThinkingBlock{Type: "thinking", Thinking: ""})
			WriteAnthropicSSE(w, flusher, "content_block_delta", converter.AnthropicContentBlockDelta{
				Type:  "content_block_delta",
				Index: blockIndex,
				Delta: converter.AnthropicThinkingDelta{Type: "thinking_delta", Thinking: chunk.Thinking},
			})
		}

		if chunk.Content != "" {
			ensureBlock("text", converter.AnthropicTextBlock{Type: "text", Text: ""})
			WriteAnthropicSSE(w, flusher, "content_block_delta", converter.AnthropicContentBlockDelta{
				Type:  "content_block_delta",
				Index: blockIndex,
				Delta: converter.AnthropicTextDelta{Type: "text_delta", Text: chunk.Content},
			})
		}

		for _, tc := range chunk.ToolCalls {
			sawToolCalls = true
			closeBlock()

			WriteAnthropicSSE(w, flusher, "content_block_start", converter.AnthropicContentBlockStart{
				Type:  "content_block_start",
				Index: blockIndex,
				ContentBlock: converter.AnthropicToolUseBlock{
					Type:  "tool_use",
					ID:    tc.ID,
					Name:  tc.Name,
					Input: json.RawMessage("{}"),
				},
			})
			WriteAnthropicSSE(w, flusher, "content_block_delta", converter.AnthropicContentBlockDelta{
				Type:  "content_block_delta",
				Index: blockIndex,
				Delta: converter.AnthropicInputJSONDelta{
					Type:        "input_json_delta",
					PartialJSON: tc.Arguments,
				},
			})
			WriteAnthropicSSE(w, flusher, "content_block_stop", converter.AnthropicContentBlockStop{
				Type:  "content_block_stop",
				Index: blockIndex,
			})
			blockIndex++
		}

		return nil
	})

	if err != nil {
		// Nothing was streamed yet, so a proper error response can still
		// be written (e.g. capability validation failures).
		if !started {
			WriteAnthropicError(w, StatusForError(err), err.Error())
		}
		return
	}
}
