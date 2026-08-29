/* ================================================================= */
/* ControlForge auto-generated C header                            */
/* Block : processor
/*
/* Implement processor_compute() in processor.c
/* Compile  : gcc -O2 -shared -fPIC -o libprocessor.so processor.c
/* ================================================================= */

#ifndef PROCESSOR_H
#define PROCESSOR_H

#ifdef __cplusplus
extern "C" {
#endif

/* -- Input struct ------------------------------------------------ */
typedef struct InputSignals {
    float sine_a;
    float gain_b;
} InputSignals;

/* -- Output struct ----------------------------------------------- */
typedef struct OutputSignals {
    float summer;
} OutputSignals;

/* -- Function signature ------------------------------------------ */
void processor_compute(const InputSignals* in, OutputSignals* out);

#ifdef __cplusplus
}
#endif

#endif /* PROCESSOR_H */