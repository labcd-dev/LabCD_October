from numpy import *

class PIDController:
    def __init__(self, kp, ki, kd, dt, output_limits=(None, None), trim=0.0):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.dt = dt
        self.limits = output_limits
        self.trim = trim
        self.N = 100.0

        self.integral = 0.0
        self.prev_error = 0.0
        self.prev_d_term = 0.0

    def update(self, setpoint, measurement):
        error = setpoint - measurement

        # Compute P and Filtered D terms
        p_term = self.kp * error
        d_term = (self.kd * self.N * (error - self.prev_error) + self.prev_d_term) / (1 + self.N * self.dt)
        self.prev_d_term = d_term
        self.prev_error = error

        # Compute unsaturated output using existing integral state
        i_term = self.ki * self.integral
        u_unsaturated = p_term + i_term + d_term + self.trim

        # Apply saturation limits
        low, high = self.limits
        output = u_unsaturated
        if low is not None or high is not None:
            output = clip(u_unsaturated, low, high)

        # --- UPDATED SIMULINK-STYLE CLAMPING ANTI-WINDUP ---

        # Calculate how far past the limit the controller wants to go
        overflow = u_unsaturated - output

        # If overflow and error share the same sign, their product is positive.
        # This means the error is pushing the output further into saturation.
        is_winding_up = (error * overflow) > 0

        # Only integrate if we are NOT winding up (or if we aren't saturated at all, where overflow == 0)
        if overflow == 0 or not is_winding_up:
            self.integral += error * self.dt

        return output