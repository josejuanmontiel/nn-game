//go:build js

package main

import (
	"log"
	"syscall/js"

	"11-juego-final/game"
)

func (g *Game) initJSBridge() {
	js.Global().Set("updateLevelConfig", js.FuncOf(func(this js.Value, args []js.Value) any {
		if len(args) < 1 {
			return nil
		}
		jsonStr := args[0].String()
		log.Printf("JS calling updateLevelConfig with data length: %d", len(jsonStr))

		if err := game.LoadLevelsFromBytes([]byte(jsonStr)); err != nil {
			log.Printf("Error loading levels from JS: %v", err)
			return err.Error()
		}

		// Reset to the first level with the new config
		g.currentLevel = game.Levels[0]
		g.ResetGame()
		g.addLog("Config updated from Web Editor")
		
		return nil
	}))
	
	log.Println("WASM JS Bridge initialized")
}
