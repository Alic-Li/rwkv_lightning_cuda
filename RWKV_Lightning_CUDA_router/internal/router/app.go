package router

import (
	"errors"
	"flag"
	"fmt"
	"log"
	"net/http"
	"strings"
	"time"

	"github.com/BurntSushi/toml"
)

// version is stamped by release builds via
// -ldflags "-X rwkv-lightning-cuda-router/internal/router.version=v<VERSION>".
var version = "dev"

// versionLine is the --version output: "<program> <version>" on one line,
// without the tag's "v" prefix so it matches the C++ binaries.
func versionLine() string {
	return "rwkv_router " + strings.TrimPrefix(version, "v")
}

func Main() {
	path := flag.String("config", "config.toml", "TOML configuration file")
	showVersion := flag.Bool("version", false, "print the version and exit")
	flag.BoolVar(showVersion, "v", false, "print the version and exit")
	flag.Parse()
	if *showVersion {
		fmt.Println(versionLine())
		return
	}
	var c config
	meta, err := toml.DecodeFile(*path, &c)
	if err != nil {
		log.Fatalf("read config: %v", err)
	}
	if unknown := meta.Undecoded(); len(unknown) > 0 {
		log.Fatalf("read config: unknown TOML key %q", unknown[0])
	}
	s, err := newScheduler(c)
	if err != nil {
		log.Fatalf("invalid config: %v", err)
	}
	policy, _ := configuredUploadPolicy(c)
	adapterPolicy, _ := configuredAdapterUploadPolicy(c)
	p := &proxy{adapterUploadPolicy: adapterPolicy, uploadPolicy: policy, scheduler: s, maxBody: c.MaxRequestBodyBytes, timeout: time.Duration(c.RequestTimeoutSeconds) * time.Second, client: &http.Client{Transport: &http.Transport{Proxy: http.ProxyFromEnvironment, MaxIdleConns: 1024, MaxIdleConnsPerHost: 256, IdleConnTimeout: 90 * time.Second, ResponseHeaderTimeout: 0, ExpectContinueTimeout: time.Second}}}
	server := &http.Server{Addr: c.Listen, Handler: p, ReadHeaderTimeout: 10 * time.Second, IdleTimeout: 120 * time.Second}
	log.Printf("RWKV router listening on %s with %d backends", c.Listen, len(s.backends))
	if err := server.ListenAndServe(); err != nil && !errors.Is(err, http.ErrServerClosed) {
		log.Fatal(err)
	}
}
