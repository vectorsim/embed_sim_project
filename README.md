# EmbedSim

**EmbedSim** is a Python-based simulation framework for developing and validating control algorithms and transferring them to embedded systems.

The framework provides a common simulation environment for **pure Python models** and **FMU-based models**, including models developed in **Modelica** and exported as FMUs.

The current control implementation is based on **Differential Flatness Control (DFC)**. The architecture is designed to allow additional control algorithms to be added in the future.

EmbedSim also includes a **C implementation targeting Infineon AURIX / TriCore**, providing a path from simulation to real-time embedded execution.

```text
System Model
   │
   ├── Python
   │
   └── FMU
        ▲
        │
     Modelica
   │
   ▼
EmbedSim
   │
   ▼
Differential Flatness Control
   │
   ▼
Embedded C
   │
   ▼
AURIX / TriCore
```

## Documentation

Detailed information about the architecture, system models, control algorithms, simulation and embedded implementation is provided in the documentation pages of this repository.

## Project Status

EmbedSim is under active development.

The framework currently provides:

* Python-based simulation
* Python and FMU-based system models
* Modelica models through FMU
* Differential Flatness Control
* Embedded C implementation
* Infineon AURIX / TriCore support

**From simulation to embedded control.**
