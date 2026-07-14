import pygame
import random

# ---------- constants ----------
SCREEN_W = 800
SCREEN_H = 600
PLAYER_W = 50
PLAYER_H = 30
BULLET_W = 6
BULLET_H = 14
ENEMY_W = 40
ENEMY_H = 30
ENEMY_BULLET_W = 6
ENEMY_BULLET_H = 14
PLAYER_SPEED = 5
BULLET_SPEED = 10
ENEMY_BULLET_SPEED = 6
ENEMY_SPEED_X = 2
ENEMY_DROP = 20
MAX_ENEMY_BULLETS = 3
ENEMY_SHOOT_DELAY = 600  # ms

# colours
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
GREEN = (0, 255, 0)
RED   = (255, 0, 0)

# ---------- classes ----------
class Player(pygame.sprite.Sprite):
    def __init__(self):
        super().__init__()
        self.image = pygame.Surface((PLAYER_W, PLAYER_H))
        self.image.fill(GREEN)
        self.rect = self.image.get_rect()
        self.rect.centerx = SCREEN_W // 2
        self.rect.bottom = SCREEN_H - 10
        self.speed = PLAYER_SPEED
        self.shoot_cooldown = 0

    def update(self, keys_pressed):
        if keys_pressed[pygame.K_LEFT]:
            self.rect.x -= self.speed
        if keys_pressed[pygame.K_RIGHT]:
            self.rect.x += self.speed
        # clamp
        if self.rect.left < 0:
            self.rect.left = 0
        if self.rect.right > SCREEN_W:
            self.rect.right = SCREEN_W

    def shoot(self, group):
        if self.shoot_cooldown <= 0:
            b = Bullet(self.rect.centerx, self.rect.top)
            group.add(b)
            self.shoot_cooldown = 15

    def update_cooldown(self):
        if self.shoot_cooldown > 0:
            self.shoot_cooldown -= 1


class Bullet(pygame.sprite.Sprite):
    def __init__(self, x, y):
        super().__init__()
        self.image = pygame.Surface((BULLET_W, BULLET_H))
        self.image.fill(WHITE)
        self.rect = self.image.get_rect()
        self.rect.centerx = x
        self.rect.bottom = y

    def update(self):
        self.rect.y -= BULLET_SPEED
        if self.rect.bottom < 0:
            self.kill()


class EnemyBullet(pygame.sprite.Sprite):
    def __init__(self, x, y):
        super().__init__()
        self.image = pygame.Surface((ENEMY_BULLET_W, ENEMY_BULLET_H))
        self.image.fill(RED)
        self.rect = self.image.get_rect()
        self.rect.centerx = x
        self.rect.top = y

    def update(self):
        self.rect.y += ENEMY_BULLET_SPEED
        if self.rect.top > SCREEN_H:
            self.kill()


class Enemy(pygame.sprite.Sprite):
    def __init__(self, x, y):
        super().__init__()
        self.image = pygame.Surface((ENEMY_W, ENEMY_H))
        self.image.fill(RED)
        self.rect = self.image.get_rect()
        self.rect.x = x
        self.rect.y = y
        self.direction = 1  # 1 right, -1 left

    def update(self, group_enemy_bullets, dt):
        self.rect.x += ENEMY_SPEED_X * self.direction
        # reverse direction on edge
        if self.rect.right >= SCREEN_W or self.rect.left <= 0:
            self.direction *= -1
            self.rect.y += ENEMY_DROP
        # random shooting
        if random.randint(0, 100) < 3 and len(group_enemy_bullets) < MAX_ENEMY_BULLETS:
            eb = EnemyBullet(self.rect.centerx, self.rect.bottom)
            group_enemy_bullets.add(eb)


# ---------- main ----------
def main():
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_W, SCREEN_H))
    pygame.display.set_caption("Space Invaders")
    clock = pygame.time.Clock()

    player = Player()
    player_group = pygame.sprite.GroupSingle(player)
    bullets = pygame.sprite.Group()
    enemy_bullets = pygame.sprite.Group()
    enemies = pygame.sprite.Group()

    # create grid of enemies
    rows = 5
    cols = 10
    for row in range(rows):
        for col in range(cols):
            x = 60 + col * 70
            y = 40 + row * 50
            enemies.add(Enemy(x, y))

    score = 0
    lives = 3
    font = pygame.font.SysFont("Arial", 24)
    running = True

    while running:
        dt = clock.tick(60) / 1000.0

        # events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    player.shoot(bullets)

        keys = pygame.key.get_pressed()
        player.update(keys)
        player.update_cooldown()

        bullets.update()
        enemy_bullets.update()

        # move enemies
        for e in enemies:
            e.update(enemy_bullets, dt)

        # collisions: bullet hits enemy
        for b in pygame.sprite.groupcollide(bullets, enemies, True, True).values():
            for _ in b:
                score += 10

        # enemy bullet hits player
        if pygame.sprite.spritecollideany(player, enemy_bullets):
            lives -= 1
            enemy_bullets.empty()
            if lives <= 0:
                running = False
            else:
                # reset player position
                player.rect.centerx = SCREEN_W // 2
                player.rect.bottom = SCREEN_H - 10

        # player bullet hits enemy bullet (optional)
        pygame.sprite.groupcollide(bullets, enemy_bullets, True, True)

        # draw
        screen.fill(BLACK)
        player_group.draw(screen)
        bullets.draw(screen)
        enemy_bullets.draw(screen)
        enemies.draw(screen)

        # score / lives
        score_text = font.render(f"Score: {score}", True, WHITE)
        lives_text = font.render(f"Lives: {lives}", True, WHITE)
        screen.blit(score_text, (10, 10))
        screen.blit(lives_text, (SCREEN_W - 100, 10))

        pygame.display.flip()

    pygame.quit()


if __name__ == "__main__":
    main()
