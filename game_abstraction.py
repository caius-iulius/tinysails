import pygame
import math

x_zero = 500
y_zero = 500
scale = 10

def draw_boat(boat, wind_vector, surf):
    def boat_poly(posx, posy, angle):
        boatpoly = [(-10,-10), (-10,10), (20,0)]
        pos = pygame.math.Vector2(posx, posy)
        rotated_points = [
            pygame.math.Vector2(x, y).rotate(angle) + pos for x, y in boatpoly]
        return rotated_points

    pygame.draw.polygon(surf, (255,0,0), boat_poly(scale*boat.position[0]+x_zero,scale*boat.position[1]+y_zero, boat.heading_angle()*180/3.14159))
    sail_angle = boat.sail_angle(wind_vector) + boat.heading_angle()
    sail_mast = (scale*boat.position[0]+x_zero+13*boat.heading[0],scale*boat.position[1]+y_zero+13*boat.heading[1])
    pygame.draw.line(surf, (255,255,255), sail_mast, (sail_mast[0] - 25*math.cos(sail_angle), sail_mast[1] - 25*math.sin(sail_angle)), 2)

def draw_buoy(buoy, surf):
    color = (0, 255, 0) if buoy.passed else (255, 0, 0)
    pygame.draw.circle(surf, color, (int(scale*buoy.position[0]+x_zero), int(scale*buoy.position[1]+y_zero)), buoy.radius*scale)


def run_game(env, get_command):
    pygame.init()
    screen = pygame.display.set_mode((1000, 1000))
    game_surf = pygame.Surface((1000, 1000))
    game_rect = pygame.Rect((0,0,1000,1000))
    clock = pygame.time.Clock()

    running = True
    last_time = pygame.time.get_ticks()
    state = env.reset()
    curr_score = 0
    while running:
        current_time = pygame.time.get_ticks()
        time_step = (current_time - last_time) / 1000.0
        last_time = current_time

        print(f"Boat state: {state}, score: {curr_score:.2f}")
        action, running = get_command(state)
        if not running:
            break
        print(f"action taken: {action}")

        state, reward, done = env.step(action, last_time, time_step)
        curr_score += reward


        game_surf.fill((0, 105, 205))
        for buoy in env.buoys:
            draw_buoy(buoy, game_surf)
        draw_boat(env.boat, env.wind_vector, game_surf)

        screen.blit(game_surf, game_rect)
        pygame.display.flip()

        if done:
            state = env.reset()
            curr_score = 0

        clock.tick(60)

    pygame.quit()