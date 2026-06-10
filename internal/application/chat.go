package application

import (
	"context"
	"fmt"

	"ollama-proxy/internal/domain"
)

// ChatUseCase is the input port that adapter handlers call.
type ChatUseCase interface {
	Chat(ctx context.Context, req *domain.ChatRequest) (*domain.ChatResponse, error)
	ChatStream(ctx context.Context, req *domain.ChatRequest, onChunk func(*domain.StreamChunk) error) error
	Generate(ctx context.Context, req *domain.GenerateRequest) (*domain.ChatResponse, error)
	GenerateStream(ctx context.Context, req *domain.GenerateRequest, onChunk func(*domain.StreamChunk) error) error
	ListModels(ctx context.Context) ([]domain.ModelInfo, error)
	Embed(ctx context.Context, model string, input []string) (*domain.EmbeddingResult, error)
}

// OllamaClient is the output port that infrastructure implements.
type OllamaClient interface {
	Chat(ctx context.Context, req *domain.ChatRequest) (*domain.ChatResponse, error)
	ChatStream(ctx context.Context, req *domain.ChatRequest) (<-chan domain.StreamChunk, error)
	Generate(ctx context.Context, req *domain.GenerateRequest) (*domain.ChatResponse, error)
	GenerateStream(ctx context.Context, req *domain.GenerateRequest) (<-chan domain.StreamChunk, error)
	ListModels(ctx context.Context) ([]domain.ModelInfo, error)
	Embed(ctx context.Context, model string, input []string) (*domain.EmbeddingResult, error)
	// ModelCapabilities reports what the model supports ("completion",
	// "vision", "tools", "thinking", "embedding", "insert", ...).
	// Returns nil when unknown, in which case validation is skipped.
	ModelCapabilities(ctx context.Context, model string) []string
}

type chatUseCase struct {
	ollama OllamaClient
}

// NewChatUseCase creates a new ChatUseCase with the given Ollama client.
func NewChatUseCase(ollama OllamaClient) ChatUseCase {
	return &chatUseCase{ollama: ollama}
}

// validateChat rejects requests the target model cannot serve, so clients
// get a clear 400 instead of an opaque backend failure. Fails open when
// capabilities are unknown (e.g. model not pulled yet).
func (uc *chatUseCase) validateChat(ctx context.Context, req *domain.ChatRequest) error {
	caps := capabilitySet(uc.ollama.ModelCapabilities(ctx, req.Model))
	if caps == nil {
		return nil
	}

	if !caps["vision"] {
		for _, m := range req.Messages {
			if len(m.Images) > 0 {
				return domain.ValidationError(fmt.Sprintf("model %q does not support image input", req.Model))
			}
		}
	}
	if !caps["tools"] && len(req.Tools) > 0 {
		return domain.ValidationError(fmt.Sprintf("model %q does not support tools", req.Model))
	}
	if !caps["thinking"] && req.Think != nil && *req.Think {
		return domain.ValidationError(fmt.Sprintf("model %q does not support thinking", req.Model))
	}
	return nil
}

func capabilitySet(caps []string) map[string]bool {
	if len(caps) == 0 {
		return nil
	}
	set := make(map[string]bool, len(caps))
	for _, c := range caps {
		set[c] = true
	}
	return set
}

func (uc *chatUseCase) Chat(ctx context.Context, req *domain.ChatRequest) (*domain.ChatResponse, error) {
	if err := uc.validateChat(ctx, req); err != nil {
		return nil, err
	}
	return uc.ollama.Chat(ctx, req)
}

func (uc *chatUseCase) ChatStream(ctx context.Context, req *domain.ChatRequest, onChunk func(*domain.StreamChunk) error) error {
	if err := uc.validateChat(ctx, req); err != nil {
		return err
	}
	chunkChan, err := uc.ollama.ChatStream(ctx, req)
	if err != nil {
		return err
	}
	for chunk := range chunkChan {
		if err := onChunk(&chunk); err != nil {
			return err
		}
		if chunk.Done {
			break
		}
	}
	return nil
}

func (uc *chatUseCase) validateGenerate(ctx context.Context, req *domain.GenerateRequest) error {
	caps := capabilitySet(uc.ollama.ModelCapabilities(ctx, req.Model))
	if caps == nil {
		return nil
	}
	if !caps["insert"] && req.Suffix != "" {
		return domain.ValidationError(fmt.Sprintf("model %q does not support suffix (fill-in-the-middle)", req.Model))
	}
	return nil
}

func (uc *chatUseCase) Generate(ctx context.Context, req *domain.GenerateRequest) (*domain.ChatResponse, error) {
	if err := uc.validateGenerate(ctx, req); err != nil {
		return nil, err
	}
	return uc.ollama.Generate(ctx, req)
}

func (uc *chatUseCase) GenerateStream(ctx context.Context, req *domain.GenerateRequest, onChunk func(*domain.StreamChunk) error) error {
	if err := uc.validateGenerate(ctx, req); err != nil {
		return err
	}
	chunkChan, err := uc.ollama.GenerateStream(ctx, req)
	if err != nil {
		return err
	}
	for chunk := range chunkChan {
		if err := onChunk(&chunk); err != nil {
			return err
		}
		if chunk.Done {
			break
		}
	}
	return nil
}

func (uc *chatUseCase) ListModels(ctx context.Context) ([]domain.ModelInfo, error) {
	return uc.ollama.ListModels(ctx)
}

func (uc *chatUseCase) Embed(ctx context.Context, model string, input []string) (*domain.EmbeddingResult, error) {
	if caps := capabilitySet(uc.ollama.ModelCapabilities(ctx, model)); caps != nil && !caps["embedding"] {
		return nil, domain.ValidationError(fmt.Sprintf("model %q does not support embeddings", model))
	}
	return uc.ollama.Embed(ctx, model, input)
}
