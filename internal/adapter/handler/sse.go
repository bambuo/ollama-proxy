package handler

import (
	"encoding/json"
	"errors"
	"fmt"
	"net/http"

	"ollama-proxy/internal/adapter/converter"
	"ollama-proxy/internal/domain"
)

const maxRequestBodySize = 10 * 1024 * 1024 // 10 MB

// StatusForError 将用例错误映射为 HTTP 状态码：验证失败是客户端的过错，其余都是服务端错误。
func StatusForError(err error) int {
	var ve domain.ValidationError
	if errors.As(err, &ve) {
		return http.StatusBadRequest
	}
	return http.StatusInternalServerError
}

// WriteSSE 将数据编码为 JSON 并作为服务器推送事件写入响应。
func WriteSSE(w http.ResponseWriter, flusher http.Flusher, data interface{}) {
	encoded, err := json.Marshal(data)
	if err != nil {
		return
	}
	logRawSSE("data", json.RawMessage(encoded))
	fmt.Fprintf(w, "data: %s\n\n", encoded)
	flusher.Flush()
}

// WriteAnthropicSSE 写入带有命名事件类型和 JSON 数据的 SSE 事件。
func WriteAnthropicSSE(w http.ResponseWriter, flusher http.Flusher, event string, data interface{}) {
	encoded, err := json.Marshal(data)
	if err != nil {
		return
	}
	logRawSSE(event, json.RawMessage(encoded))
	fmt.Fprintf(w, "event: %s\ndata: %s\n\n", event, encoded)
	flusher.Flush()
}

// WriteError 发送结构化的 OpenAI 格式错误响应。
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

// WriteAnthropicError 发送 Anthropic 格式的错误响应。
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
