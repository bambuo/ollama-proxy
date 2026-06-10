package converter

import (
	"crypto/rand"
	"encoding/hex"
	"strings"
	"time"
)

// IDTimestamp returns a unique timestamp suitable for OpenAI-style IDs.
func IDTimestamp() int64 {
	return time.Now().UnixNano()
}

// NowUnix returns the current Unix timestamp.
func NowUnix() int64 {
	return time.Now().Unix()
}

// FormatHex returns a random hex string suitable for Anthropic-style message IDs.
func FormatHex() string {
	b := make([]byte, 12)
	if _, err := rand.Read(b); err != nil {
		return "0000000000000000000000ff"
	}
	return hex.EncodeToString(b)
}

// NormalizeStop converts a stop value (string or array of strings) to []string.
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

// ExtractBase64Image strips a data-URL prefix (e.g. "data:image/png;base64,")
// from an image URL, returning the raw base64 payload. Returns "" for
// non-data URLs, which the local Ollama backend cannot fetch.
func ExtractBase64Image(url string) string {
	if !strings.HasPrefix(url, "data:") {
		return ""
	}
	if idx := strings.Index(url, "base64,"); idx >= 0 {
		return url[idx+len("base64,"):]
	}
	return ""
}

// IsRemoteURL reports whether s is an http(s) URL.
func IsRemoteURL(s string) bool {
	return strings.HasPrefix(s, "http://") || strings.HasPrefix(s, "https://")
}

// EstimateTokens gives a rough token count for a text (≈4 chars per token).
func EstimateTokens(text string) int {
	n := len(text) / 4
	if n < 1 && len(text) > 0 {
		n = 1
	}
	return n
}
