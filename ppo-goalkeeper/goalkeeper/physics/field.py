"""Field geometry, in metres, in each environment's local frame.

x points from the goal toward the penalty spot, y runs along the goal line, z is up.
The goal line's rear edge (the posts' back face) is the plane x = 0; the goal opens toward +x.
Sizes follow futsal: a 3 m x 2 m goal with the penalty mark 6 m out.
"""

# Ball: a size-5 football (Poly Haven "football" model is 22.2 cm across).
BALL_RADIUS = 0.11
BALL_MASS = 0.43

# Goal (Objaverse futsal goal, scaled so the inside of the posts is exactly 3.0 m apart).
GOAL_HALF_WIDTH = 1.5        # inner edge of each post
GOAL_HEIGHT = 2.02           # underside of the crossbar
POST_DEPTH = 0.082           # along x
POST_WIDTH = 0.117           # along y
NET_DEPTH = 1.09             # back of the net, behind the goal line

PENALTY_X = 6.0

# Goalkeeper: a Clearpath Jackal driving sideways along the goal line, with a blocking panel facing the kicker.
KEEPER_X = 0.45              # base_link x in front of the goal line
KEEPER_Y_LIMIT = 1.08        # soft travel limit of the base centre (panel edge stays 4.8 cm inside the posts)
PANEL_HALF_WIDTH = 0.372     # half-width of the blocking panel along y (0.70 m sheet + 2.2 cm frame each side)
PANEL_OFFSET_X = 0.28        # panel front face, in front of base_link
PANEL_FRONT_X = KEEPER_X + PANEL_OFFSET_X
# Plane where the ball centre meets the panel; the interception point is predicted on this plane.
CONTACT_X = PANEL_FRONT_X + BALL_RADIUS

# Largest |y| at the contact plane that the panel can still cover from the travel limit.
KEEPER_COVER = KEEPER_Y_LIMIT + PANEL_HALF_WIDTH + BALL_RADIUS
