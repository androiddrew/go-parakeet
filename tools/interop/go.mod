module github.com/androiddrew/go-parakeet/tools/interop

go 1.25.0

require (
	github.com/androiddrew/go-g2p v0.2.0
	github.com/androiddrew/go-kokoro v0.2.0
	github.com/androiddrew/go-parakeet v0.0.0
	github.com/yalue/onnxruntime_go v1.22.0
)

require (
	github.com/androiddrew/ortenv v0.2.0 // indirect
	github.com/dlclark/regexp2 v1.11.5 // indirect
	golang.org/x/text v0.41.0 // indirect
)

replace github.com/androiddrew/go-parakeet => ../..
