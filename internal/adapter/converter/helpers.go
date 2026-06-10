package converter

import (
	"crypto/rand"
	"encoding/hex"
	"strings"
	"time"
)

// IDTimestamp 返回适用于 OpenAI 风格 ID 的唯一时间戳。
func IDTimestamp() int64 {
	return time.Now().UnixNano()
}

// NowUnix 返回当前的 Unix 时间戳。
func NowUnix() int64 {
	return time.Now().Unix()
}

// FormatHex 返回适用于 Anthropic 风格消息 ID 的随机十六进制字符串。
func FormatHex() string {
	b := make([]byte, 12)
	if _, err := rand.Read(b); err != nil {
		return "0000000000000000000000ff"
	}
	return hex.EncodeToString(b)
}

// NormalizeStop 将停止值（字符串或字符串数组）转换为 []string。
func NormalizeStop(stop interface{}) []string {
	switch v := stop.(type) {
	case string:
		if v == "" {
			return nil
		}
		return []string{v}
	case []interface{}:
		var result []string
		for _, s := range v {
			if str, ok := s.(string); ok {
				result = append(result, str)
			}
		}
		return result
	default:
		return nil
	}
}

// ExtractBase64Image 从图片 URL 中去除 data-URL 前缀（例如 "data:image/png;base64,"），
// 返回原始 base64 负载。对于非 data URL 返回 ""，因为本地 Ollama 后端无法获取这些 URL。
func ExtractBase64Image(url string) string {
	if !strings.HasPrefix(url, "data:") {
		return ""
	}
	if idx := strings.Index(url, "base64,"); idx >= 0 {
		return url[idx+len("base64,"):]
	}
	return ""
}

// IsRemoteURL 报告 s 是否为 http(s) URL。
func IsRemoteURL(s string) bool {
	return strings.HasPrefix(s, "http://") || strings.HasPrefix(s, "https://")
}

// EstimateTokens 粗略估算文本的 token 数量（约 4 字符/token）。
func EstimateTokens(text string) int {
	n := len(text) / 4
	if n < 1 && len(text) > 0 {
		n = 1
	}
	return n
}
