#include "three_phase_processor.h"

void three_phase_processor_compute(const InputSignals* in, OutputSignals* out) {
    // Your C implementation here
    out->gain[0] = in->source[0] * 2.0;
    out->gain[1] = in->source[1] * 2.0;
    out->gain[2] = in->source[2] * 2.0;
}