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
		fmt.Printf("✅ Ollama Proxy running at http://127.0.0.1%s\n", addr)
		fmt.Println("   Listening for requests...")
		if err := server.ListenAndServe(); err != nil && err != http.ErrServerClosed {
			fmt.Fprintf(os.Stderr, "❌ Server error: %v\n", err)
			os.Exit(1)
		}
	}()

	// 阻塞直到收到信号
	sig := <-quit
	fmt.Printf("\n🛑 Received signal %v, shutting down gracefully...\n", sig)

	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	if err := server.Shutdown(ctx); err != nil {
		fmt.Fprintf(os.Stderr, "❌ Server forced to shutdown: %v\n", err)
		os.Exit(1)
	}

	fmt.Println("✅ Server stopped")
}
