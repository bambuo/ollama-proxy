package handler

import (
	"bytes"
	"encoding/json"
	"fmt"
)

func logRawRequest(body []byte) {
	fmt.Println(string(compactJSON(body)))
}

func logRawResponse(data interface{}) {
	encoded, err := json.Marshal(data)
	if err != nil {
		fmt.Printf(">> 响应 (序列化失败: %v)\n", err)
		return
	}
	fmt.Printf(">> 响应: %s\n", string(compactJSON(encoded)))
}

func logRawSSE(event string, data json.RawMessage) {
	fmt.Printf(">> SSE [%s]: %s\n", event, string(compactJSON(data)))
}

func compactJSON(data []byte) json.RawMessage {
	var buf bytes.Buffer
	if err := json.Compact(&buf, data); err != nil {
		return json.RawMessage(data)
	}
	return json.RawMessage(buf.Bytes())
}
