package main

import (
	"context"
	"fmt"
	"net/http"
	"os"
	"os/signal"
	"syscall"
	"time"

	"ollama-proxy/internal/adapter/handler"
	"ollama-proxy/internal/application"
	"ollama-proxy/internal/infrastructure"
)

func main() {
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
