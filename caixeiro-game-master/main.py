import pygame
import sys
import os
from ui.manager import UIManager

def main():
    # Centraliza a janela na tela
    os.environ['SDL_VIDEO_CENTERED'] = '1'
    
    print("Iniciando Pygame...")
    pygame.init()
    
    WIDTH, HEIGHT = 1280, 720
    print(f"Configurando modo de vídeo {WIDTH}x{HEIGHT}...")
    
    # Usa DOUBLEBUF para evitar cintilação e problemas de renderização em alguns drivers Linux
    screen = pygame.display.set_mode((WIDTH, HEIGHT), pygame.DOUBLEBUF) 
    pygame.display.set_caption("FEI - TSP Solver: GA vs ACO")
    
    print("Inicializando UIManager...")
    clock = pygame.time.Clock()
    ui_manager = UIManager(screen)
    
    print("Entrando no loop principal...")
    frame_count = 0
    running = True
    while running:
        # Processamento de Eventos
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.VIDEORESIZE:
                WIDTH, HEIGHT = event.w, event.h
                screen = pygame.display.set_mode((WIDTH, HEIGHT), pygame.RESIZABLE | pygame.DOUBLEBUF)
                ui_manager.on_resize(WIDTH, HEIGHT)
                
            ui_manager.handle_event(event)
            
        # Atualização da Lógica
        ui_manager.update()
        
        # Desenho da Interface
        ui_manager.draw()
        
        # Log a cada 60 frames (aprox. 1 segundo) para confirmar que não travou
        frame_count += 1
        if frame_count % 60 == 0:
            print(f"Simulador rodando... (Frame: {frame_count})", end="\r")
            
        clock.tick(60)

    pygame.quit()
    print("\nSimulador encerrado.")

if __name__ == "__main__":
    main()
