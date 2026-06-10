package handler

import (
	"fmt"
	"net/http"
	"strings"

	"ollama-proxy/internal/application"
)

// Server 是顶层 HTTP 处理器，将请求路由到相应的处理器。
type Server struct {
	openai    *OpenAIHandler
	anthropic *AnthropicHandler
}

// New 通过依赖注入创建一个新的 Server。
func New(chatUC application.ChatUseCase) *Server {
	return &Server{
		openai:    NewOpenAIHandler(chatUC),
		anthropic: NewAnthropicHandler(chatUC),
	}
}

// ServeHTTP 实现了 http.Handler 接口。
func (s *Server) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	path := r.URL.Path

	switch path {
	case "/v1/chat/completions", "/chat/completions":
		s.openai.Handle(w, r)
		return
	case "/v1/completions", "/completions":
		s.openai.HandleCompletions(w, r)
		return
	case "/v1/models", "/models":
		s.openai.HandleModels(w, r)
		return
	case "/v1/embeddings", "/embeddings":
		s.openai.HandleEmbeddings(w, r)
		return
	case "/v1/messages":
		s.anthropic.Handle(w, r)
		return
	case "/v1/messages/count_tokens":
		s.anthropic.HandleCountTokens(w, r)
		return
	}

	if modelID, ok := strings.CutPrefix(path, "/v1/models/"); ok && modelID != "" {
		s.openai.HandleModel(w, r, modelID)
		return
	}

	WriteError(w, http.StatusNotFound, fmt.Sprintf("path not found: %s", path))
}
