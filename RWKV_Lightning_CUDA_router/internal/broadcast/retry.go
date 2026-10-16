package broadcast

import (
	"context"
	"crypto/rand"
	"net/http"
	"strconv"
	"time"
)

type Policy struct {
	Attempts       int
	AttemptTimeout time.Duration
	BaseDelay      time.Duration
	MaxDelay       time.Duration
}

func (p Policy) Defaults() Policy {
	if p.Attempts == 0 {
		p.Attempts = 3
	}
	if p.AttemptTimeout == 0 {
		p.AttemptTimeout = 120 * time.Second
	}
	if p.BaseDelay == 0 {
		p.BaseDelay = 250 * time.Millisecond
	}
	if p.MaxDelay == 0 {
		p.MaxDelay = 2 * time.Second
	}
	return p
}

func (p Policy) Delay(attempt int) time.Duration {
	delay := p.BaseDelay
	for i := 0; i < attempt && delay < p.MaxDelay; i++ {
		delay *= 2
	}
	if delay > p.MaxDelay {
		delay = p.MaxDelay
	}
	var random [1]byte
	if _, err := rand.Read(random[:]); err == nil {
		delay = delay/2 + time.Duration(random[0])*delay/256
	}
	if delay > p.MaxDelay {
		delay = p.MaxDelay
	}
	return delay
}

func RetryAfter(header http.Header) time.Duration {
	value := header.Get("Retry-After")
	if seconds, err := strconv.ParseInt(value, 10, 32); err == nil && seconds > 0 {
		return time.Duration(seconds) * time.Second
	}
	if date, err := http.ParseTime(value); err == nil && time.Until(date) > 0 {
		return time.Until(date)
	}
	return 0
}

func waitRetry(ctx context.Context, delay time.Duration) bool {
	timer := time.NewTimer(delay)
	defer timer.Stop()
	select {
	case <-ctx.Done():
		return false
	case <-timer.C:
		return true
	}
}

func Retryable(status int, err error) bool {
	return err != nil || status == 408 || status == 429 || status == 500 || status == 502 || status == 503 || status == 504
}

func (p Policy) Budget(retry bool) time.Duration {
	p = p.Defaults()
	if !retry {
		return p.AttemptTimeout
	}
	return time.Duration(p.Attempts)*p.AttemptTimeout + time.Duration(p.Attempts-1)*p.MaxDelay
}
