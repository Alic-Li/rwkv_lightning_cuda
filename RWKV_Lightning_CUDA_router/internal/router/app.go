package router

import (
	"errors"
	"flag"
	"log"
	"net/http"
	"time"

	"github.com/BurntSushi/toml"
)

func Main() {
	path := flag.String("config", "config.toml", "TOML configuration file")
	flag.Parse()
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
