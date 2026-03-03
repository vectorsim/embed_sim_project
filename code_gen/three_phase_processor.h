/* ================================================================= */
/* ControlForge auto-generated C header                            */
/* Block : three_phase_processor
/*
/* Implement three_phase_processor_compute() in three_phase_processor.c
/* Compile  : gcc -O2 -shared -fPIC -o libthree_phase_processor.so three_phase_processor.c
/* ================================================================= */

#ifndef THREE_PHASE_PROCESSOR_H
#define THREE_PHASE_PROCESSOR_H

#ifdef __cplusplus
extern "C" {
#endif

/* -- Input struct ------------------------------------------------ */
typedef struct InputSignals {
    double source[3];
} InputSignals;

/* -- Output struct ----------------------------------------------- */
typedef struct OutputSignals {
    double gain[3];
} OutputSignals;

/* -- Function signature ------------------------------------------ */
void three_phase_processor_compute(const InputSignals* in, OutputSignals* out);

#ifdef __cplusplus
}
#endif

#endif /* THREE_PHASE_PROCESSOR_H */