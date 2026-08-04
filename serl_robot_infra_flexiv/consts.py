ACTION_SCALE = [0.05, 1.0]
LEFT_ARM_SERIAL = "Rizon 4s-063533"
RIGHT_ARM_SERIAL = "Rizon 4s-063440"

KZ = 500.0 # z axis linear stiffness,  in N/m. lower = springier. the nominal value is 10k
BZ = 30.0 # z axis damping, in N*s/m - opposes z velocity so the spring settles instead of bouncing.
          # conservative starting point: raise it if it's still bouncy, lower it if it feels sluggish/sticky

MAX_LIN_VEL = 0.05 # m/s
MAX_ANG_VEL = 0.3 # rad/s ~17deg per second