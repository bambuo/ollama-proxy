package application

import (
	"context"

	"ollama-proxy/internal/domain"
)

// ChatUseCase is the input port that adapter handlers call.
type ChatUseCase interface {
	Chat(ctx context.Context, req *domain.ChatRequest) (*domain.ChatResponse, error)
	ChatStream(ctx context.Context, req *domain.ChatRequest, onChunk func(*domain.StreamChunk) error) error
	ListModels(ctx context.Context) ([]domain.ModelInfo, error)
	Embed(ctx context.Context, model string, input []string) (*domain.EmbeddingResult, error)
}

// OllamaClient is the output port that infrastructure implements.
type OllamaClient interface {
	Chat(ctx context.Context, req *domain.ChatRequest) (*domain.ChatResponse, error)
	ChatStream(ctx context.Context, req *domain.ChatRequest) (<-chan domain.StreamChunk, error)
	ListModels(ctx context.Context) ([]domain.ModelInfo, error)
	Embed(ctx context.Context, model string, input []string) (*domain.EmbeddingResult, error)
}

type chatUseCase struct {
	ollama OllamaClient
}

// NewChatUseCase creates a new ChatUseCase with the given Ollama client.
func NewChatUseCase(ollama OllamaClient) ChatUseCase {
	return &chatUseCase{ollama: ollama}
}

func (uc *chatUseCase) Chat(ctx context.Context, req *domain.ChatRequest) (*domain.ChatResponse, error) {
	return uc.ollama.Chat(ctx, req)
}

func (uc *chatUseCase) ChatStream(ctx context.Context, req *domain.ChatRequest, onChunk func(*domain.StreamChunk) error) error {
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

func (uc *chatUseCase) ListModels(ctx context.Context) ([]domain.ModelInfo, error) {
	return uc.ollama.ListModels(ctx)
}

func (uc *chatUseCase) Embed(ctx context.Context, model string, input []string) (*domain.EmbeddingResult, error) {
	return uc.ollama.Embed(ctx, model, input)
}
