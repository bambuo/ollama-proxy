package handler

import (
	"encoding/json"
	"fmt"
	"net/http"

	"ollama-proxy/internal/adapter/converter"
)

const maxRequestBodySize = 10 * 1024 * 1024 // 10 MB

// WriteSSE marshals data as JSON and writes it as a Server-Sent Event.
func WriteSSE(w http.ResponseWriter, flusher http.Flusher, data interface{}) {
	encoded, err := json.Marshal(data)
	if err != nil {
		return
	}
	fmt.Fprintf(w, "data: %s\n\n", encoded)
	flusher.Flush()
}

// WriteAnthropicSSE writes an SSE event with a named event type and JSON data.
func WriteAnthropicSSE(w http.ResponseWriter, flusher http.Flusher, event string, data interface{}) {
	encoded, err := json.Marshal(data)
	if err != nil {
		return
	}
	fmt.Fprintf(w, "event: %s\ndata: %s\n\n", event, encoded)
	flusher.Flush()
}

// WriteError sends a structured OpenAI-format error response.
func WriteError(w http.ResponseWriter, status int, msg string) {
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

	errBody, _ := json.Marshal(converter.OpenAIErrorResponse{
		Error: converter.OpenAIError{
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

// WriteAnthropicError sends an Anthropic-format error response.
func WriteAnthropicError(w http.ResponseWriter, status int, msg string) {
	errType := "server_error"
	switch {
	case status == http.StatusBadRequest, status == http.StatusMethodNotAllowed:
		errType = "invalid_request_error"
	case status == http.StatusNotFound:
		errType = "not_found_error"
	case status >= 500:
		errType = "server_error"
	}

	errBody, _ := json.Marshal(converter.AnthropicErrorResp{
		Type: "error",
		Error: converter.AnthropicErr{
			Type:    errType,
			Message: msg,
		},
	})

	w.Header().Set("Content-Type", "application/json; charset=utf-8")
	w.WriteHeader(status)
	w.Write(errBody)
}
