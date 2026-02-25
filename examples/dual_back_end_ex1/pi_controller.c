 /* pi_controller.c
   * Hand-written C implementation of the PI controller.
   * Compile together with the Cython wrapper via setup_pi_controller.py.
   */
  #include "pi_controller.h"
  
  static float _Kp = 1.0f;
  static float _Ki = 0.0f;
  
  void pi_controller_init(float Kp, float Ki) {
      _Kp = Kp;
      _Ki = Ki;
  }
  
  void pi_controller_compute(const InputSignals* in,
                                   OutputSignals* out,
                                   StateSignals*  state,
                                   float dt) {
      /* Euler integration of the integral term */
      state->integral      += _Ki * in->error * dt;
      out->control_output   = _Kp * in->error + state->integral;
  }
  
  void pi_controller_reset(StateSignals* state) {
      state->integral = 0.0f;
  }
