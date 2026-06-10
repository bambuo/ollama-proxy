package handler

import (
	"encoding/json"
	"fmt"
	"io"
	"net/http"

	"ollama-proxy/internal/adapter/converter"
	"ollama-proxy/internal/application"
	"ollama-proxy/internal/domain"
)

// OpenAIHandler handles OpenAI-compatible chat completion requests.
type OpenAIHandler struct {
	uc application.ChatUseCase
}

// NewOpenAIHandler creates a new OpenAIHandler.
func NewOpenAIHandler(uc application.ChatUseCase) *OpenAIHandler {
	return &OpenAIHandler{uc: uc}
}

// Handle processes OpenAI /v1/chat/completions requests.
func (h *OpenAIHandler) Handle(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		w.Header().Set("Allow", "POST")
		WriteError(w, http.StatusMethodNotAllowed, "Only POST method is allowed")
		return
	}

	body, err := io.ReadAll(http.MaxBytesReader(w, r.Body, maxRequestBodySize))
	if err != nil {
		WriteError(w, http.StatusBadRequest, err.Error())
		return
	}

	var openaiReq converter.OpenAIChatRequest
	if err := json.Unmarshal(body, &openaiReq); err != nil {
		WriteError(w, http.StatusBadRequest, err.Error())
		return
	}

	if openaiReq.Model == "" {
		WriteError(w, http.StatusBadRequest, "model is required")
		return
	}

	domainReq := converter.OpenAIRequestToDomain(openaiReq)

	if domainReq.Stream {
		h.handleStream(w, r, &domainReq, &openaiReq)
	} else {
		h.handleNonStream(w, r, &domainReq)
	}
}

// HandleModels processes GET /v1/models requests.
func (h *OpenAIHandler) HandleModels(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		w.Header().Set("Allow", "GET")
		WriteError(w, http.StatusMethodNotAllowed, "Only GET method is allowed")
		return
	}

	models, err := h.uc.ListModels(r.Context())
	if err != nil {
		WriteError(w, http.StatusInternalServerError, err.Error())
		return
	}

	w.Header().Set("Content-Type", "application/json; charset=utf-8")
	json.NewEncoder(w).Encode(converter.DomainToOpenAIModelList(models))
}

// HandleModel processes GET /v1/models/{id} requests.
func (h *OpenAIHandler) HandleModel(w http.ResponseWriter, r *http.Request, modelID string) {
	if r.Method != http.MethodGet {
		w.Header().Set("Allow", "GET")
		WriteError(w, http.StatusMethodNotAllowed, "Only GET method is allowed")
		return
	}

	models, err := h.uc.ListModels(r.Context())
	if err != nil {
		WriteError(w, http.StatusInternalServerError, err.Error())
		return
	}

	for _, m := range models {
		if m.Name == modelID {
			w.Header().Set("Content-Type", "application/json; charset=utf-8")
			json.NewEncoder(w).Encode(converter.OpenAIModel{
				ID:      m.Name,
				Object:  "model",
				Created: converter.NowUnix(),
				OwnedBy: "ollama",
			})
			return
		}
	}
	WriteError(w, http.StatusNotFound, fmt.Sprintf("model %q not found", modelID))
}

// HandleEmbeddings processes POST /v1/embeddings requests.
func (h *OpenAIHandler) HandleEmbeddings(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		w.Header().Set("Allow", "POST")
		WriteError(w, http.StatusMethodNotAllowed, "Only POST method is allowed")
		return
	}

	body, err := io.ReadAll(http.MaxBytesReader(w, r.Body, maxRequestBodySize))
	if err != nil {
		WriteError(w, http.StatusBadRequest, err.Error())
		return
	}

	var embReq converter.OpenAIEmbeddingRequest
	if err := json.Unmarshal(body, &embReq); err != nil {
		WriteError(w, http.StatusBadRequest, err.Error())
		return
	}

	if embReq.Model == "" {
		WriteError(w, http.StatusBadRequest, "model is required")
		return
	}

	var input []string
	switch v := embReq.Input.(type) {
	case string:
		input = []string{v}
	case []interface{}:
		for _, item := range v {
			s, ok := item.(string)
			if !ok {
				WriteError(w, http.StatusBadRequest, "input must be a string or an array of strings")
				return
			}
			input = append(input, s)
		}
	default:
		WriteError(w, http.StatusBadRequest, "input must be a string or an array of strings")
		return
	}
	if len(input) == 0 {
		WriteError(w, http.StatusBadRequest, "input is required")
		return
	}

	result, err := h.uc.Embed(r.Context(), embReq.Model, input)
	if err != nil {
		WriteError(w, http.StatusInternalServerError, err.Error())
		return
	}

	w.Header().Set("Content-Type", "application/json; charset=utf-8")
	json.NewEncoder(w).Encode(converter.DomainToOpenAIEmbeddings(embReq.Model, result))
}

func (h *OpenAIHandler) handleNonStream(w http.ResponseWriter, r *http.Request, domainReq *domain.ChatRequest) {
	resp, err := h.uc.Chat(r.Context(), domainReq)
	if err != nil {
		WriteError(w, http.StatusInternalServerError, err.Error())
		return
	}

	openaiResp := converter.DomainToOpenAIResponse(domainReq.Model, resp)

	w.Header().Set("Content-Type", "application/json; charset=utf-8")
	json.NewEncoder(w).Encode(openaiResp)
}

func (h *OpenAIHandler) handleStream(w http.ResponseWriter, r *http.Request, domainReq *domain.ChatRequest, openaiReq *converter.OpenAIChatRequest) {
	// SSE headers
	w.Header().Set("Content-Type", "text/event-stream; charset=utf-8")
	w.Header().Set("Cache-Control", "no-cache")
	w.Header().Set("Connection", "keep-alive")
	w.Header().Set("X-Accel-Buffering", "no")

	flusher, ok := w.(http.Flusher)
	if !ok {
		WriteError(w, http.StatusInternalServerError, "streaming not supported")
		return
	}

	includeUsage := openaiReq.StreamOptions != nil && openaiReq.StreamOptions.IncludeUsage
	id := fmt.Sprintf("chatcmpl-%d", converter.IDTimestamp())
	created := converter.NowUnix()

	toolCallIndex := 0
	sawToolCalls := false

	err := h.uc.ChatStream(r.Context(), domainReq, func(chunk *domain.StreamChunk) error {
		// Check client disconnect
		select {
		case <-r.Context().Done():
			return r.Context().Err()
		default:
		}

		if len(chunk.ToolCalls) > 0 {
			sawToolCalls = true
		}

		openaiChunk, send := converter.BuildOpenAIStreamChunk(id, created, domainReq.Model, *chunk, &toolCallIndex, sawToolCalls)
		if send {
			WriteSSE(w, flusher, openaiChunk)
		}

		if chunk.Done {
			if includeUsage {
				usageChunk := converter.BuildOpenAIUsageChunk(id, created, domainReq.Model, chunk.InputTokens, chunk.OutputTokens)
				WriteSSE(w, flusher, usageChunk)
			}
			fmt.Fprintf(w, "data: [DONE]\n\n")
			flusher.Flush()
		}
		return nil
	})

	if err != nil {
		// If we already sent headers and partial data, we can't write an error response.
		// The error is likely from client disconnect, which is expected.
		return
	}
}
