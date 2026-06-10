package handler

import (
	"encoding/json"
	"fmt"
	"io"
	"net/http"

	"ollama-proxy/internal/adapter/converter"
	"ollama-proxy/internal/domain"
)

// HandleCompletions 处理旧版 OpenAI /v1/completions 请求，
// 由 Ollama 的 /api/generate 支持。
func (h *OpenAIHandler) HandleCompletions(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		w.Header().Set("Allow", "POST")
		WriteError(w, http.StatusMethodNotAllowed, "仅允许 POST 方法")
		return
	}

	body, err := io.ReadAll(http.MaxBytesReader(w, r.Body, maxRequestBodySize))
	if err != nil {
		WriteError(w, http.StatusBadRequest, err.Error())
		return
	}

	var completionReq converter.OpenAICompletionRequest
	if err := json.Unmarshal(body, &completionReq); err != nil {
		WriteError(w, http.StatusBadRequest, err.Error())
		return
	}

	if completionReq.Model == "" {
		WriteError(w, http.StatusBadRequest, "必须指定模型")
		return
	}

	domainReq, ok := converter.CompletionRequestToDomain(completionReq)
	if !ok {
		WriteError(w, http.StatusBadRequest, "prompt 必须是字符串（不支持多个 prompt）")
		return
	}

	if domainReq.Stream {
		h.handleCompletionStream(w, r, &domainReq, &completionReq)
	} else {
		h.handleCompletionNonStream(w, r, &domainReq)
	}
}

func (h *OpenAIHandler) handleCompletionNonStream(w http.ResponseWriter, r *http.Request, domainReq *domain.GenerateRequest) {
	resp, err := h.uc.Generate(r.Context(), domainReq)
	if err != nil {
		WriteError(w, StatusForError(err), err.Error())
		return
	}

	w.Header().Set("Content-Type", "application/json; charset=utf-8")
	json.NewEncoder(w).Encode(converter.DomainToOpenAICompletion(domainReq.Model, resp))
}

func (h *OpenAIHandler) handleCompletionStream(w http.ResponseWriter, r *http.Request, domainReq *domain.GenerateRequest, completionReq *converter.OpenAICompletionRequest) {
	// SSE 响应头
	w.Header().Set("Content-Type", "text/event-stream; charset=utf-8")
	w.Header().Set("Cache-Control", "no-cache")
	w.Header().Set("Connection", "keep-alive")
	w.Header().Set("X-Accel-Buffering", "no")

	flusher, ok := w.(http.Flusher)
	if !ok {
		WriteError(w, http.StatusInternalServerError, "不支持流式传输")
		return
	}

	includeUsage := completionReq.StreamOptions != nil && completionReq.StreamOptions.IncludeUsage
	id := fmt.Sprintf("cmpl-%d", converter.IDTimestamp())
	created := converter.NowUnix()

	wroteAny := false
	err := h.uc.GenerateStream(r.Context(), domainReq, func(chunk *domain.StreamChunk) error {
		select {
		case <-r.Context().Done():
			return r.Context().Err()
		default:
		}
		wroteAny = true

		completionChunk, send := converter.BuildOpenAICompletionChunk(id, created, domainReq.Model, *chunk)
		if send {
			WriteSSE(w, flusher, completionChunk)
		}

		if chunk.Done {
			if includeUsage {
				usageChunk := converter.OpenAICompletionResponse{
					ID:      id,
					Object:  "text_completion",
					Created: created,
					Model:   domainReq.Model,
					Choices: []converter.OpenAICompletionChoice{},
					Usage: &converter.OpenAIUsage{
						PromptTokens:     chunk.InputTokens,
						CompletionTokens: chunk.OutputTokens,
						TotalTokens:      chunk.InputTokens + chunk.OutputTokens,
					},
				}
				WriteSSE(w, flusher, usageChunk)
			}
			fmt.Fprintf(w, "data: [DONE]\n\n")
			flusher.Flush()
		}
		return nil
	})

	if err != nil {
		if !wroteAny {
			WriteError(w, StatusForError(err), err.Error())
		}
		return
	}
}
