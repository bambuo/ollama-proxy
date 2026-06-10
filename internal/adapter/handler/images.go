package handler

import (
	"context"
	"encoding/base64"
	"fmt"
	"io"
	"net/http"
	"time"

	"ollama-proxy/internal/adapter/converter"
	"ollama-proxy/internal/domain"
)

const maxImageSize = 20 * 1024 * 1024 // 20 MB

var imageHTTPClient = &http.Client{Timeout: 30 * time.Second}

// resolveRemoteImages 下载转换器留在请求中的任何 http(s) 图片引用，
// 并将其替换为 base64 负载，因为 Ollama 只接受内联 base64 图片。
func resolveRemoteImages(ctx context.Context, req *domain.ChatRequest) error {
	for i := range req.Messages {
		for j, img := range req.Messages[i].Images {
			if !converter.IsRemoteURL(img) {
				continue
			}
			encoded, err := fetchImageBase64(ctx, img)
			if err != nil {
				return fmt.Errorf("获取图片 %s：%w", img, err)
			}
			req.Messages[i].Images[j] = encoded
		}
	}
	return nil
}

func fetchImageBase64(ctx context.Context, url string) (string, error) {
	httpReq, err := http.NewRequestWithContext(ctx, http.MethodGet, url, nil)
	if err != nil {
		return "", err
	}

	resp, err := imageHTTPClient.Do(httpReq)
	if err != nil {
		return "", err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return "", fmt.Errorf("状态码 %d", resp.StatusCode)
	}

	data, err := io.ReadAll(io.LimitReader(resp.Body, maxImageSize+1))
	if err != nil {
		return "", err
	}
	if len(data) > maxImageSize {
		return "", fmt.Errorf("图片超过 %d 字节限制", maxImageSize)
	}

	return base64.StdEncoding.EncodeToString(data), nil
}
