#include "three_phase_processor.h"

void three_phase_processor_compute(const InputSignals* in, OutputSignals* out) {
    // Your C implementation here
    out->source[0] = in->source[0] * 2.0;
    out->source[1] = in->source[1] * 2.0;
    out->source[2] = in->source[2] * 2.0;
}