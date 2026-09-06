import time
import math
import numpy as np
from abc import ABC, abstractmethod

class AbstractViewer(ABC):

    def __init__(self):
        pass

    @abstractmethod
    def display(self, state: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def close(self):
        pass

class viewer3d(AbstractViewer):
    # Do we need it??
    pass

class viewer2D(AbstractViewer):
    def __init__(self, render_fps: int = 30, render_mode: str = "human"): 
        self.render_fps = render_fps
        self.render_mode = render_mode
        self.screen_height = 400
        self.screen_width = 900
        self.screen = None
        self.clock = None

    def display(self, state: np.ndarray) -> np.ndarray:

        import pygame
        from pygame import gfxdraw

        if self.screen is None:
            pygame.init()
            if self.render_mode == "human":
                pygame.display.init()
                pygame.display.set_caption('Ball Balancing on Arc')
                self.screen = pygame.display.set_mode(
                    (self.screen_width, self.screen_height)
                )
            else:  # mode == "rgb_array"
                self.screen = pygame.Surface((self.screen_width, self.screen_height))

        if self.clock is None:
            self.clock = pygame.time.Clock()

        world_width = 2.4 * 2
        scale = self.screen_width / world_width
        ballradius = 3
        cartwidth = 150.0
        cartheight = 50.0

        if state is None:
            return None

        x = state

        # Plain floats: pygame's Rect rejects numpy float32 scalars
        cart_position = float(x[0])  # Position of the cart
        theta = float(x[2])          # Angle of the arc (angle of the ball)
        cartx = cart_position * scale + self.screen_width / 2.0 - cartwidth/2  # MIDDLE OF CART

        cart = pygame.Rect((cartx, self.screen_height*0.7-cartheight/2, cartwidth, cartheight))

        # Define the rectangle position and size
        rect_x, rect_y, rect_width, rect_height = cartx, self.screen_height*0.7-cartheight/2, cartwidth, cartheight

        # Define the arc parameters
        arc_rect = (cartx, rect_y + rect_height-7, rect_width, 15)

        self.surf = pygame.Surface((self.screen_width, self.screen_height))
        self.surf.fill((255, 255, 255))

        # Draw the three-sided cart (top, left, and right borders)
            # Draw carriage
        pygame.draw.rect(self.surf, (0,0,0), cart, 2)
        # Draw the white border for the top side
        pygame.draw.line(self.surf, (255,255,255), (rect_x, rect_y), (rect_x + rect_width, rect_y ), 2)

        # Draw Arc on Carriage
        pygame.draw.arc(self.surf, (0,0,0), arc_rect, 0, math.pi, 1) 

        # Calculate the center of the arc
        arc_center_x = arc_rect[0] + arc_rect[2] / 2
        arc_center_y = arc_rect[1] + arc_rect[3] / 2

        # Calculate the position of the ball
        ball_x = arc_center_x + (arc_rect[2] / 2) * math.cos(math.pi*3/2+theta)
        ball_y = arc_center_y + (arc_rect[3] / 2) * math.sin(math.pi*3/2-theta) - ballradius 


        # Draw the ball
        gfxdraw.filled_circle(self.surf, int(ball_x), int(ball_y), int(ballradius), (0, 0, 0))


        pygame.draw.line(self.surf, (0,0,0), (0, self.screen_height*0.76),\
            (self.screen_width, self.screen_height*0.76))


        self.screen.blit(self.surf, (0, 0))
        if self.render_mode == "human":
            pygame.event.pump()
            self.clock.tick(self.render_fps)
            pygame.display.flip()
        
        return np.transpose(
                np.array(pygame.surfarray.pixels3d(self.screen)), axes=(1, 0, 2)
            )

    def close(self):
        if self.screen is not None:
            import pygame
            
            pygame.display.quit()
            pygame.quit()