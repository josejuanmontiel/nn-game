.PHONY: build build-gpu build-wasm serve clean

build:
	go build -o simulator .

build-gpu:
	go build -tags gpu -o simulator_gpu .

build-wasm:
	GOOS=js GOARCH=wasm go build -o web/game.wasm .

serve:
	@echo "Starting local server at http://localhost:8080/web/index.html"
	python3 -m http.server 8080

clean:
	rm -f simulator simulator_gpu web/game.wasm game/11-juego-final
