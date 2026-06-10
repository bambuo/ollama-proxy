package main

import (
	"bufio"
	"context"
	"fmt"
	"net/http"
	"os"
	"os/signal"
	"strings"
	"syscall"
	"time"

	"ollama-proxy/internal/adapter/handler"
	"ollama-proxy/internal/application"
	"ollama-proxy/internal/infrastructure"
)

func main() {
	loadEnv(".env")

	addr := os.Getenv("PORT")
	if addr == "" {
		addr = ":3000"
	} else if addr[0] != ':' {
		addr = ":" + addr
	}

	mux := http.NewServeMux()

	ollamaClient := infrastructure.NewOllamaClient()
	chatUC := application.NewChatUseCase(ollamaClient)
	srv := handler.New(chatUC)

	mux.Handle("/", srv)

	server := &http.Server{
		Addr:    addr,
		Handler: mux,
	}

	// 用于监听操作系统信号的通道
	quit := make(chan os.Signal, 1)
	signal.Notify(quit, syscall.SIGINT, syscall.SIGTERM)

	go func() {
		fmt.Printf("✅ Ollama 代理运行在 http://127.0.0.1%s\n", addr)
		fmt.Println("   正在监听请求...")
		if err := server.ListenAndServe(); err != nil && err != http.ErrServerClosed {
			fmt.Fprintf(os.Stderr, "❌ 服务器错误：%v\n", err)
			os.Exit(1)
		}
	}()

	// 阻塞直到收到信号
	sig := <-quit
	fmt.Printf("\n🛑 收到信号 %v，正在优雅关闭...\n", sig)

	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	if err := server.Shutdown(ctx); err != nil {
		fmt.Fprintf(os.Stderr, "❌ 服务器强制关闭：%v\n", err)
		os.Exit(1)
	}

	fmt.Println("✅ 服务器已停止")
}

// loadEnv 加载 .env 文件中的环境变量（无第三方依赖）。
// 仅设置尚未由操作系统设置的环境变量，因此操作系统环境变量优先级更高。
func loadEnv(path string) {
	f, err := os.Open(path)
	if err != nil {
		return // .env 不存在时静默跳过
	}
	defer f.Close()

	scanner := bufio.NewScanner(f)
	for scanner.Scan() {
		line := strings.TrimSpace(scanner.Text())
		// 跳过空行和注释
		if line == "" || strings.HasPrefix(line, "#") {
			continue
		}
		// 支持 export KEY=VALUE 语法
		line = strings.TrimPrefix(line, "export ")

		name, val, ok := strings.Cut(line, "=")
		if !ok {
			continue
		}
		name = strings.TrimSpace(name)
		val = strings.TrimSpace(val)
		// 移除可选的引号
		if len(val) >= 2 && ((val[0] == '"' && val[len(val)-1] == '"') || (val[0] == '\'' && val[len(val)-1] == '\'')) {
			val = val[1 : len(val)-1]
		}
		// 不覆盖已存在的环境变量
		if os.Getenv(name) == "" {
			os.Setenv(name, val)
		}
	}
}
