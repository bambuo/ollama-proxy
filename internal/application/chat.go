package application

import (
	"context"
	"fmt"

	"ollama-proxy/internal/domain"
)

// ChatUseCase 是适配器处理器调用的输入端口。
type ChatUseCase interface {
	Chat(ctx context.Context, req *domain.ChatRequest) (*domain.ChatResponse, error)
	ChatStream(ctx context.Context, req *domain.ChatRequest, onChunk func(*domain.StreamChunk) error) error
	Generate(ctx context.Context, req *domain.GenerateRequest) (*domain.ChatResponse, error)
	GenerateStream(ctx context.Context, req *domain.GenerateRequest, onChunk func(*domain.StreamChunk) error) error
	ListModels(ctx context.Context) ([]domain.ModelInfo, error)
	Embed(ctx context.Context, model string, input []string) (*domain.EmbeddingResult, error)
}

// OllamaClient 是基础设施实现的输出端口。
type OllamaClient interface {
	Chat(ctx context.Context, req *domain.ChatRequest) (*domain.ChatResponse, error)
	ChatStream(ctx context.Context, req *domain.ChatRequest) (<-chan domain.StreamChunk, error)
	Generate(ctx context.Context, req *domain.GenerateRequest) (*domain.ChatResponse, error)
	GenerateStream(ctx context.Context, req *domain.GenerateRequest) (<-chan domain.StreamChunk, error)
	ListModels(ctx context.Context) ([]domain.ModelInfo, error)
	Embed(ctx context.Context, model string, input []string) (*domain.EmbeddingResult, error)
	// ModelCapabilities 报告模型支持的能力（"completion"、"vision"、"tools"、"thinking"、"embedding"、"insert" 等）。
	// 未知时返回 nil，此时跳过验证。
	ModelCapabilities(ctx context.Context, model string) []string
}

type chatUseCase struct {
	ollama OllamaClient
}

// NewChatUseCase 使用给定的 Ollama 客户端创建一个新的 ChatUseCase。
func NewChatUseCase(ollama OllamaClient) ChatUseCase {
	return &chatUseCase{ollama: ollama}
}

// validateChat 拒绝目标模型无法服务的请求，以便客户端获得清晰的 400 错误而非模糊的后端失败。
// 当能力未知时（例如模型尚未拉取），放行通过。
func (uc *chatUseCase) validateChat(ctx context.Context, req *domain.ChatRequest) error {
	caps := capabilitySet(uc.ollama.ModelCapabilities(ctx, req.Model))
	if caps == nil {
		return nil
	}

	if !caps["vision"] {
		for _, m := range req.Messages {
			if len(m.Images) > 0 {
				return domain.ValidationError(fmt.Sprintf("模型 %q 不支持图片输入", req.Model))
			}
		}
	}
	if !caps["tools"] && len(req.Tools) > 0 {
		return domain.ValidationError(fmt.Sprintf("模型 %q 不支持工具", req.Model))
	}
	if !caps["thinking"] && req.Think != nil && *req.Think {
		return domain.ValidationError(fmt.Sprintf("模型 %q 不支持推理", req.Model))
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
		return domain.ValidationError(fmt.Sprintf("模型 %q 不支持后缀（中间填充）", req.Model))
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
		return nil, domain.ValidationError(fmt.Sprintf("模型 %q 不支持嵌入向量", model))
	}
	return uc.ollama.Embed(ctx, model, input)
}
