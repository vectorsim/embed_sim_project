# EmbedSim

**PMSM Fachschale Motor Development and Embedded Control Simulation**

**EmbedSim** is a Python-based development and simulation environment for **PMSM (Permanent Magnet Synchronous Motor) Fachschale development**.

It provides a common environment for developing and validating PMSM motor models and control algorithms, with support for **pure Python models** and **FMU-based models**, including models developed in **Modelica** and exported as FMUs.

The current control implementation is based on **Differential Flatness Control (DFC)**. The architecture is designed to allow additional control algorithms to be implemented in the future.

EmbedSim also includes a **C implementation targeting Infineon AURIX / TriCore**, providing a path from PMSM simulation and control development to real-time embedded execution.

```text
                         PMSM Fachschale
                              │
                 ┌────────────┴────────────┐
                 │                         │
                 ▼                         ▼
          Pure Python Model              FMU
                                           ▲
                                           │
                                        Modelica
                 │                         │
                 └────────────┬────────────┘
                              │
                              ▼
                         EmbedSim
                       Simulation
                              │
                              ▼
                 Differential Flatness
                       Control (DFC)
                              │
                              ▼
                        Embedded C
                              │
                              ▼
                     AURIX / TriCore
```

## Documentation

Detailed information about the **PMSM Fachschale**, system models, simulation environment, Differential Flatness Control and AURIX implementation is provided in the documentation pages of this repository.

## Project Status

EmbedSim is under active development.

Currently implemented:

* PMSM Fachschale motor development
* Python-based simulation
* Pure Python system models
* FMU-based system models
* Modelica models through FMU
* Differential Flatness Control (DFC)
* Embedded C implementation
* Infineon AURIX / TriCore target

Additional control algorithms can be implemented within the framework in the future.

**PMSM development — from model and simulation to embedded control.**
