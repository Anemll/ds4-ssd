import pygame
import random
import math

# ---------- constants ----------
SCREEN_W = 800
SCREEN_H = 600
FPS = 60

# colours
BLACK   = (0,   0,   0)
WHITE   = (255, 255, 255)
GREEN   = (0,   255, 0)
RED     = (255,  0,   0)
YELLOW  = (255, 255,  0)
CYAN    = (0,   255, 255)

# ---------- player ----------
PLAYER_W = 50
PLAYER_H = 30
PLAYER_SPEED = 5
PLAYER_LIVES = 3

# ---------- aliens ----------
ALIEN_W = 40
ALIEN_H = 30
ALIEN_SPEED_X = 2
ALIEN_DROP_Y = 20
ALIEN_ROWS = 5
ALIEN_COLS = 10
ALIEN_GAP = 10

# ---------- bullets ----------
BULLET_W = 4
BULLET_H = 12
BULLET_SPEED = 10
ALIEN_BULLET_SPEED = 6

# ---------- explosion ----------
EXPLOSION_FRAMES = 10


# ======================================================================
# Sprite helpers
# ======================================================================
def load_surf(w, h, colour):
    s = pygame.Surface((w, h))
    s.fill(colour)
    return s


# ======================================================================
# Game class
# ======================================================================
class SpaceInvaders:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((SCREEN_W, SCREEN_H))
        pygame.display.set_caption("Space Invaders")
        self.clock = pygame.time.Clock()
        self.running = True
        self.reset()

    # ----- reset / init -----
    def reset(self):
        self.score = 0
        self.lives = PLAYER_LIVES
        self.level = 1
        self.game_over = False
        self.won = False

        # player
        self.player_rect = pygame.Rect(
            SCREEN_W // 2 - PLAYER_W // 2,
            SCREEN_H - 60,
            PLAYER_W, PLAYER_H,
        )
        self.player_surf = load_surf(PLAYER_W, PLAYER_H, GREEN)

        # bullets
        self.bullets = []          # player bullets
        self.alien_bullets = []    # alien bullets

        # aliens
        self.alien_grid = []
        self.alien_direction = 1   # 1 = right, -1 = left
        self.alien_move_timer = 0
        self.alien_move_delay = 20  # frames between moves
        self.alien_fire_timer = 0
        self.alien_fire_delay = 60  # frames between alien shots

        self._build_alien_grid()

        # explosions
        self.explosions = []       # list of (rect, frame)

        # shield blocks
        self.shields = []
        self._build_shields()

    def _build_alien_grid(self):
        self.alien_grid.clear()
        start_x = (SCREEN_W - (ALIEN_COLS * (ALIEN_W + ALIEN_GAP))) // 2
        start_y = 50
        for row in range(ALIEN_ROWS):
            for col in range(ALIEN_COLS):
                x = start_x + col * (ALIEN_W + ALIEN_GAP)
                y = start_y + row * (ALIEN_H + ALIEN_GAP)
                rect = pygame.Rect(x, y, ALIEN_W, ALIEN_H)
                # alternate colours per row
                colour = RED if row % 2 == 0 else YELLOW
                surf = load_surf(ALIEN_W, ALIEN_H, colour)
                self.alien_grid.append({
                    'rect': rect,
                    'surf': surf,
                    'alive': True,
                })

    def _build_shields(self):
        # three shield blocks near bottom
        for i in range(3):
            x = 150 + i * 200
            y = SCREEN_H - 120
            w = 60
            h = 20
            rect = pygame.Rect(x, y, w, h)
            surf = load_surf(w, h, CYAN)
            self.shields.append({'rect': rect, 'surf': surf, 'hp': 3})

    # ----- input -----
    def handle_events(self):
        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                self.running = False
            elif e.type == pygame.KEYDOWN:
                if e.key == pygame.K_SPACE and not self.game_over and not self.won:
                    self._fire_bullet()
                elif e.key == pygame.K_r and (self.game_over or self.won):
                    self.reset()

    def _fire_bullet(self):
        bx = self.player_rect.centerx - BULLET_W // 2
        by = self.player_rect.top - BULLET_H
        rect = pygame.Rect(bx, by, BULLET_W, BULLET_H)
        self.bullets.append({'rect': rect, 'surf': load_surf(BULLET_W, BULLET_H, WHITE)})

    # ----- update -----
    def update(self):
        if self.game_over or self.won:
            return

        # --- player movement ---
        keys = pygame.key.get_pressed()
        dx = 0
        if keys[pygame.K_LEFT]:
            dx = -PLAYER_SPEED
        if keys[pygame.K_RIGHT]:
            dx = PLAYER_SPEED
        self.player_rect.x += dx
        self.player_rect.clamp_ip(self.screen.get_rect())

        # --- move bullets ---
        for b in self.bullets[:]:
            b['rect'].y -= BULLET_SPEED
            if b['rect'].bottom < 0:
                self.bullets.remove(b)
        for b in self.alien_bullets[:]:
            b['rect'].y += ALIEN_BULLET_SPEED
            if b['rect'].top > SCREEN_H:
                self.alien_bullets.remove(b)

        # --- alien movement ---
        self.alien_move_timer += 1
        if self.alien_move_timer >= self.alien_move_delay:
            self.alien_move_timer = 0
            # move all aliens
            move_x = ALIEN_SPEED_X * self.alien_direction
            move_y = 0
            # check edge
            for a in self.alien_grid:
                if not a['alive']:
                    continue
                if a['rect'].right + move_x > SCREEN_W or a['rect'].left + move_x < 0:
                    move_x = 0
                    move_y = ALIEN_DROP_Y
                    self.alien_direction *= -1
                    break
            for a in self.alien_grid:
                if not a['alive']:
                    continue
                a['rect'].x += move_x
                a['rect'].y += move_y

        # --- alien firing ---
        self.alien_fire_timer += 1
        if self.alien_fire_timer >= self.alien_fire_delay:
            self.alien_fire_timer = 0
            alive = [a for a in self.alien_grid if a['alive']]
            if alive:
                shooter = random.choice(alive)
                r = shooter['rect']
                bx = r.centerx - BULLET_W // 2
                by = r.bottom
                rect = pygame.Rect(bx, by, BULLET_W, BULLET_H)
                self.alien_bullets.append({
                    'rect': rect,
                    'surf': load_surf(BULLET_W, BULLET_H, RED),
                })

        # --- collision: player bullets vs aliens ---
        for b in self.bullets[:]:
            for a in self.alien_grid:
                if not a['alive']:
                    continue
                if b['rect'].colliderect(a['rect']):
                    a['alive'] = False
                    self.bullets.remove(b)
                    self.score += 10 * self.level
                    self._add_explosion(a['rect'].center)
                    break

        # --- collision: alien bullets vs player ---
        for b in self.alien_bullets[:]:
            if b['rect'].colliderect(self.player_rect):
                self.alien_bullets.remove(b)
                self._hit_player()
                break

        # --- collision: alien bullets vs shields ---
        for b in self.alien_bullets[:]:
            for s in self.shields[:]:
                if b['rect'].colliderect(s['rect']):
                    self.alien_bullets.remove(b)
                    s['hp'] -= 1
                    if s['hp'] <= 0:
                        self.shields.remove(s)
                    break

        # --- collision: player bullets vs shields ---
        for b in self.bullets[:]:
            for s in self.shields[:]:
                if b['rect'].colliderect(s['rect']):
                    self.bullets.remove(b)
                    s['hp'] -= 1
                    if s['hp'] <= 0:
                        self.shields.remove(s)
                    break

        # --- collision: aliens vs player ---
        for a in self.alien_grid:
            if not a['alive']:
                continue
            if a['rect'].colliderect(self.player_rect):
                a['alive'] = False
                self._hit_player()
                break

        # --- check win / lose ---
        alive_count = sum(1 for a in self.alien_grid if a['alive'])
        if alive_count == 0:
            self.won = True
            self.score += 500 * self.level

        # --- update explosions ---
        self.explosions = [
            (r, f + 1) for r, f in self.explosions if f + 1 < EXPLOSION_FRAMES
        ]

    def _hit_player(self):
        self.lives -= 1
        self._add_explosion(self.player_rect.center)
        if self.lives <= 0:
            self.game_over = True
        else:
            # reposition player
            self.player_rect.x = SCREEN_W // 2 - PLAYER_W // 2

    def _add_explosion(self, center):
        w, h = 30, 30
        r = pygame.Rect(center[0] - w // 2, center[1] - h // 2, w, h)
        self.explosions.append((r, 0))

    # ----- draw -----
    def draw(self):
        self.screen.fill(BLACK)

        # draw player
        if not self.game_over:
            self.screen.blit(self.player_surf, self.player_rect)

        # draw aliens
        for a in self.alien_grid:
            if a['alive']:
                self.screen.blit(a['surf'], a['rect'])

        # draw bullets
        for b in self.bullets:
            self.screen.blit(b['surf'], b['rect'])
        for b in self.alien_bullets:
            self.screen.blit(b['surf'], b['rect'])

        # draw shields
        for s in self.shields:
            self.screen.blit(s['surf'], s['rect'])

        # draw explosions
        for rect, frame in self.explosions:
            # fade out
            alpha = 255 - int(255 * frame / EXPLOSION_FRAMES)
            colour = (255, alpha, 0)
            pygame.draw.circle(self.screen, colour, rect.center, rect.w // 2)

        # draw UI
        font = pygame.font.Font(None, 36)
        score_text = font.render(f"Score: {self.score}", True, WHITE)
        self.screen.blit(score_text, (10, 10))
        lives_text = font.render(f"Lives: {self.lives}", True, WHITE)
        self.screen.blit(lives_text, (SCREEN_W - 120, 10))
        level_text = font.render(f"Level: {self.level}", True, WHITE)
        self.screen.blit(level_text, (SCREEN_W // 2 - 40, 10))

        # game over / win overlay
        if self.game_over:
            go_text = font.render("GAME OVER - Press R to restart", True, RED)
            tw = go_text.get_width()
            self.screen.blit(go_text, ((SCREEN_W - tw) // 2, SCREEN_H // 2))
        elif self.won:
            win_text = font.render("YOU WIN! - Press R to restart", True, GREEN)
            tw = win_text.get_width()
            self.screen.blit(win_text, ((SCREEN_W - tw) // 2, SCREEN_H // 2))

        pygame.display.flip()

    # ----- main loop -----
    def run(self):
        while self.running:
            self.handle_events()
            self.update()
            self.draw()
            self.clock.tick(FPS)
        pygame.quit()


# ======================================================================
# entry
# ======================================================================
if __name__ == "__main__":
    game = SpaceInvaders()
    game.run()
