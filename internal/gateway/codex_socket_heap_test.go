package gateway

import (
	"fmt"
	"net/http"
	"runtime"
	"strconv"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/coder/websocket"

	"github.com/yetone/magpie/internal/provider"
)

// A long Codex conversation's turn is tens of MB (one session sent
// 90 MB a turn, its reply beginning ~50 s later), and its bytes were held
// in four copies while the backend thought: as read, as the
// response.create message, every input item again for the next turn's
// delta, and on a kept socket the whole message beside the delta, in case
// a new socket had to take it. The message goes once it is written, the
// items are held as their digests, and a kept socket's turn is read again
// from the request should it go whole: the turn as read is all a turn
// holds of its own while its reply is awaited, a fresh socket's or a kept
// one's.
func TestCodexSocketTurnLetsItsBytesGo(t *testing.T) {
	socketOn(t)
	defer provider.CodexZstdForTests(true)()
	got := make(chan struct{})
	release := make(chan struct{})
	chatgpt(t, func(w http.ResponseWriter, r *http.Request) {
		c, err := websocket.Accept(w, r, nil)
		if err != nil {
			return
		}
		defer c.CloseNow()
		c.SetReadLimit(-1)
		for n := 1; ; n++ {
			if _, _, err := c.Read(r.Context()); err != nil {
				return
			}
			// the message read is let go here, as the backend's would be:
			// what is held from now on is magpie's
			got <- struct{}{}
			<-release
			id := "resp-" + strconv.Itoa(n)
			for _, fr := range []string{
				`{"type":"response.created","response":{"id":"` + id + `"}}`,
				`{"type":"response.output_item.done","output_index":0,"item":{"id":"msg_a","type":"message","status":"completed","role":"assistant","content":[{"type":"output_text","text":"pong"}]}}`,
				`{"type":"response.completed","response":{"id":"` + id + `","usage":{"input_tokens":7,"output_tokens":1}}}`,
			} {
				if c.Write(r.Context(), websocket.MessageText, []byte(fr)) != nil {
					return
				}
			}
		}
	})

	var history []string
	for i := range 64 {
		history = append(history, fmt.Sprintf(`{"type":"message","role":"user","content":[{"type":"input_text","text":%q}]}`,
			strings.Repeat(fmt.Sprintf("line %d of a long tool output, ", i), 8000)))
	}
	for _, c := range []struct {
		name string
		body string
	}{
		// the first turn sets up what every turn after it shares
		{"warm-up", keptTurn("conv-0", "gpt-5.5", userPing)},
		{"fresh socket", keptTurn("conv-1", "gpt-5.5", history...)},
		{"kept socket", keptTurn("conv-1", "gpt-5.5", append(history, asstPong, userAgain)...)},
	} {
		before := settledHeap()
		var wg sync.WaitGroup
		wg.Go(func() { keptPost(t, "acct-1", c.body) })
		<-got
		held := settledHeap() - before
		release <- struct{}{}
		wg.Wait()
		if times := float64(held) / float64(len(c.body)); c.name != "warm-up" && times > 2 {
			t.Errorf("%s: a %d MB turn holds %d MB awaiting its reply, %.1f times the turn", c.name, len(c.body)>>20, held>>20, times)
		}
	}
}

// settledHeap is the live heap once it stops moving: the last turn's
// goroutines, and the write of this one's message, have ended.
func settledHeap() int64 {
	var ms runtime.MemStats
	last := int64(-1)
	for range 50 {
		runtime.GC()
		runtime.ReadMemStats(&ms)
		now := int64(ms.HeapAlloc)
		if d := now - last; last >= 0 && d < 1<<20 && d > -1<<20 {
			return now
		}
		last = now
		time.Sleep(10 * time.Millisecond)
	}
	return last
}
