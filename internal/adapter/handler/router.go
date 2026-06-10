package handler

import (
	"fmt"
	"net/http"
	"strings"

	"ollama-proxy/internal/application"
)

// Server is the top-level HTTP handler that routes requests to the appropriate handler.
type Server struct {
	openai    *OpenAIHandler
	anthropic *AnthropicHandler
}

// New creates a new Server with dependency injection.
func New(chatUC application.ChatUseCase) *Server {
	return &Server{
		openai:    NewOpenAIHandler(chatUC),
		anthropic: NewAnthropicHandler(chatUC),
	}
}

// ServeHTTP implements http.Handler.
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
