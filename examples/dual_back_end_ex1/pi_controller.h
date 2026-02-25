/* pi_controller.h
   * Auto-generated C header for EmbedSim block 'pi_controller'
   * Implement pi_controller_compute() in pi_controller.c
   * Compile: gcc -O2 -shared -fPIC -o libpi_controller.so pi_controller.c
   */
  #ifndef PI_CONTROLLER_H
  #define PI_CONTROLLER_H

  typedef struct InputSignals  { float error;          } InputSignals;
  typedef struct OutputSignals { float control_output; } OutputSignals;
  typedef struct StateSignals  { float integral;        } StateSignals;

  void pi_controller_init   (float Kp, float Ki);
  void pi_controller_compute(const InputSignals* in,
                                   OutputSignals* out,
                                   StateSignals*  state,
                                   float dt);
  void pi_controller_reset  (StateSignals* state);

  #endif /* PI_CONTROLLER_H */
