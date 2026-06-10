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

// resolveRemoteImages downloads any http(s) image references left in the
// request by the converters and replaces them with base64 payloads, since
// Ollama only accepts inline base64 images.
func resolveRemoteImages(ctx context.Context, req *domain.ChatRequest) error {
	for i := range req.Messages {
		for j, img := range req.Messages[i].Images {
			if !converter.IsRemoteURL(img) {
				continue
			}
			encoded, err := fetchImageBase64(ctx, img)
			if err != nil {
				return fmt.Errorf("fetch image %s: %w", img, err)
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
		return "", fmt.Errorf("status %d", resp.StatusCode)
	}

	data, err := io.ReadAll(io.LimitReader(resp.Body, maxImageSize+1))
	if err != nil {
		return "", err
	}
	if len(data) > maxImageSize {
		return "", fmt.Errorf("image exceeds %d bytes", maxImageSize)
	}

	return base64.StdEncoding.EncodeToString(data), nil
}
