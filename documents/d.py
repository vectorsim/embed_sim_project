"""
Complete EmbedSim Documentation PDF Generator
Run: pip install fpdf
Run: python create_embedsim_pdf.py
"""

from fpdf import FPDF
import datetime

class PDF(FPDF):
    def header(self):
        self.set_font('Arial', 'B', 15)
        self.cell(80)
        self.cell(30, 10, 'EmbedSim Framework', 0, 0, 'C')
        self.ln(20)
    
    def footer(self):
        self.set_y(-15)
        self.set_font('Arial', 'I', 8)
        self.cell(0, 10, f'Page {self.page_no()}', 0, 0, 'C')
    
    def chapter_title(self, title):
        self.set_font('Arial', 'B', 14)
        self.set_fill_color(200, 220, 240)
        self.cell(0, 10, title, 0, 1, 'L', 1)
        self.ln(4)
    
    def section_title(self, title):
        self.set_font('Arial', 'B', 12)
        self.set_text_color(0, 0, 0)
        self.cell(0, 8, title, 0, 1, 'L')
        self.ln(2)
    
    def subsection_title(self, title):
        self.set_font('Arial', 'B', 11)
        self.set_text_color(60, 60, 60)
        self.cell(0, 6, title, 0, 1, 'L')
        self.ln(1)
    
    def chapter_body(self, text):
        self.set_font('Arial', '', 10)
        self.multi_cell(0, 5, text)
        self.ln()
    
    def code_block(self, code):
        self.set_font('Courier', '', 8)
        self.set_fill_color(245, 245, 245)
        self.set_text_color(0, 0, 0)
        self.multi_cell(0, 4, code, 0, 'L', 1)
        self.ln(3)
    
    def diagram(self, diagram_text):
        self.set_font('Courier', '', 8)
        self.set_fill_color(230, 240, 250)
        self.multi_cell(0, 4, diagram_text, 0, 'L', 1)
        self.ln(3)
    
    def bullet_point(self, text):
        self.set_font('Arial', '', 10)
        self.cell(5)
        self.cell(0, 5, f"• {text}", 0, 1)

# Create PDF
pdf = PDF()
pdf.add_page()
pdf.set_auto_page_break(auto=True, margin=15)

# Title Page
pdf.set_font('Arial', 'B', 28)
pdf.set_text_color(0, 51, 102)
pdf.cell(0, 60, 'EmbedSim Framework', 0, 1, 'C')
pdf.set_font('Arial', 'B', 18)
pdf.set_text_color(0, 0, 0)
pdf.cell(0, 10, 'Complete Architecture and User Guide', 0, 1, 'C')
pdf.set_font('Arial', '', 12)
pdf.cell(0, 30, f'Version 3.0.0', 0, 1, 'C')
pdf.set_font('Arial', 'I', 10)
pdf.cell(0, 10, f'Generated: {datetime.datetime.now().strftime("%B %d, %Y")}', 0, 1, 'C')

# Table of Contents
pdf.add_page()
pdf.chapter_title('Table of Contents')

toc_content = """
1. INTRODUCTION .................................................. 3
   1.1 What is EmbedSim? ........................................ 3
   1.2 Key Features ............................................. 3
   1.3 System Requirements ...................................... 3

2. OVERALL ARCHITECTURE ......................................... 4
   2.1 High-Level Architecture Diagram .......................... 4
   2.2 Core Design Philosophy ................................... 4
   2.3 Component Layers ......................................... 5

3. HOW EMBEDSIM WORKS ........................................... 6
   3.1 The Execution Flow ....................................... 6
   3.2 Phase 1: Block Creation .................................. 7
   3.3 Phase 2: Connection Building ............................. 7
   3.4 Phase 3: Graph Traversal ................................. 8
   3.5 Phase 4: Time-Stepping Simulation ........................ 9
   3.6 The Compute Cycle Deep Dive .............................. 10
   3.7 Dynamic Block Integration ................................ 12
   3.8 Feedback Loop Handling ................................... 14

4. BLOCK REFERENCE .............................................. 16
   4.1 Source Blocks ............................................ 16
   4.2 Processing Blocks ........................................ 18
   4.3 Dynamic Blocks ........................................... 20
   4.4 Special Blocks ........................................... 22

5. SIMULATION ENGINE ............................................ 24
   5.1 The EmbedSim Class ....................................... 24
   5.2 Signal Recording with VectorScope ........................ 25
   5.3 Running Simulations ...................................... 26
   5.4 Visualization ............................................ 27

6. CODE GENERATION .............................................. 28
   6.1 The Code Generation Pipeline ............................. 28
   6.2 Marking Code Generation Regions .......................... 29
   6.3 Generated Files .......................................... 30
   6.4 Compilation and Usage .................................... 32

7. ADVANCED FEATURES ............................................ 33
   7.1 Feedback Loops and Loop Breakers ......................... 33
   7.2 ScriptBlock Deep Dive .................................... 35
   7.3 FMU Co-Simulation ........................................ 37
   7.4 Custom Block Creation .................................... 39

8. EXAMPLES ..................................................... 41
   8.1 First Simulation ......................................... 41
   8.2 PID Controller ........................................... 42
   8.3 Three-Phase Inverter Control ............................. 44
   8.4 Sensor Fusion with Kalman Filter ......................... 46

9. API REFERENCE ................................................ 48
   9.1 Core Classes ............................................. 48
   9.2 Key Functions ............................................ 49
   9.3 Block Parameters ......................................... 50

10. TROUBLESHOOTING ............................................ 51
    10.1 Common Issues .......................................... 51
    10.2 Debugging Tips ......................................... 52
    10.3 Performance Optimization ............................... 53

11. VERSION HISTORY ............................................. 54
"""
pdf.chapter_body(toc_content)

# ======================================================================
# CHAPTER 1: INTRODUCTION
# ======================================================================
pdf.add_page()
pdf.chapter_title('1. INTRODUCTION')

pdf.section_title('1.1 What is EmbedSim?')
intro1 = """
EmbedSim is a lightweight block-diagram simulation framework designed specifically 
for 32-bit embedded platforms. It provides a MATLAB/Simulink-like experience in 
Python, with the unique ability to generate production-ready C code from the same 
block diagrams used during development.

The framework bridges the gap between algorithm development and embedded deployment,
allowing engineers to:
• Develop and test control algorithms in Python
• Simulate complex dynamic systems with feedback loops
• Generate optimized C code for microcontrollers
• Co-simulate with FMU models from other tools
• Deploy the same algorithm with zero code changes
"""
pdf.chapter_body(intro1)

pdf.section_title('1.2 Key Features')
pdf.chapter_body("")
pdf.bullet_point("Float32-First Design: Optimized for MCU compatibility (ARM Cortex-M, Infineon AURIX, etc.)")
pdf.bullet_point("Dual Backend: Same blocks work in Python (development) and C (production)")
pdf.bullet_point("Vector-Based: All signals are vectors, enabling multi-channel processing")
pdf.bullet_point("Feedback Loop Support: Automatic detection and handling of algebraic loops")
pdf.bullet_point("Code Generation: Generate C headers, Cython wrappers, and SimBlock stubs")
pdf.bullet_point("FMU Integration: Co-simulation with Functional Mock-up Units (FMI standard)")
pdf.bullet_point("Script-to-C: Write Python, automatically transpile to C with AST analysis")
pdf.bullet_point("Multiple ODE Solvers: Euler (fast), RK4 (accurate), Heun (compromise)")
pdf.chapter_body("")

pdf.section_title('1.3 System Requirements')
reqs = """
• Python 3.8 or higher
• NumPy 1.20 or higher
• Matplotlib (for plotting and visualization)
• C compiler: gcc, clang, or MSVC (for C backend)
• Cython (for C wrapper compilation)
• Optional: FMPy (for FMU support)
• Optional: pytest (for running tests)
"""
pdf.chapter_body(reqs)

# ======================================================================
# CHAPTER 2: OVERALL ARCHITECTURE
# ======================================================================
pdf.add_page()
pdf.chapter_title('2. OVERALL ARCHITECTURE')

pdf.section_title('2.1 High-Level Architecture Diagram')
diagram = """
┌─────────────────────────────────────────────────────────────────────────────┐
│                           EMBEDSIM FRAMEWORK                                │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                         USER APPLICATION                             │   │
│  │  (Builds block diagrams, configures simulations, generates code)     │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                    │                                         │
│                                    ▼                                         │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                        PUBLIC API (__init__.py)                      │   │
│  │  • Block classes          • Simulation engine     • Code generators  │   │
│  │  • Utility functions      • Version info          • Type definitions │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                    │                                         │
│         ┌──────────────────────────┼──────────────────────────┐            │
│         ▼                          ▼                          ▼            │
│  ┌───────────────┐         ┌───────────────┐         ┌───────────────┐    │
│  │  BLOCK LAYER  │         │    ENGINE     │         │   GENERATOR   │    │
│  │               │         │    LAYER      │         │    LAYER      │    │
│  │ • Core Blocks │         │ • Simulation  │         │ • C Code Gen  │    │
│  │ • Sources     │◄────────┤   Engine      │────────►│ • Cython Wraps│    │
│  │ • Processing  │         │ • Loop        │         │ • Headers     │    │
│  │ • Dynamic     │────────►│   Detection   │         │ • Setup Script│    │
│  │ • Script      │         │ • Scope       │         └───────────────┘    │
│  │ • FMU         │         │ • Statistics  │                 │            │
│  └───────────────┘         └───────────────┘                 │            │
│         │                          │                          │            │
│         ▼                          ▼                          ▼            │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                        EXECUTION TARGETS                              │   │
│  ├─────────────────────┬─────────────────────┬─────────────────────────┤   │
│  │   Python Runtime    │   C/C++ Backend     │   FMU Co-simulation     │   │
│  │   (Development)     │   (Production)      │   (Model Exchange)      │   │
│  └─────────────────────┴─────────────────────┴─────────────────────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
"""
pdf.diagram(diagram)

pdf.section_title('2.2 Core Design Philosophy')
philosophy = """
EmbedSim is built on five fundamental principles:

1. Vector-First Architecture: Every signal is a vector, enabling multi-channel 
   processing without redundant code. A single block can process 3-phase currents,
   position-velocity states, or any multi-dimensional signal.

2. Dual Backend Design: Same blocks work in Python (development) and C (production). 
   Switch between backends with a single boolean flag - no code changes required.

3. Explicit Connections: Block diagrams are built using intuitive >> operators,
   making the signal flow visually clear and self-documenting.

4. Automatic Topological Ordering: The engine analyzes the block graph and 
   determines execution order automatically. No manual sorting required.

5. Loop-Aware Simulation: Handles feedback loops through special "breaker" blocks
   that provide previous values, preventing algebraic loops while maintaining
   causal behavior.
"""
pdf.chapter_body(philosophy)

pdf.section_title('2.3 Component Layers')
layers = """
The framework is organized into four main layers:

LAYER 1: USER APPLICATION
• Where users build block diagrams
• Configure simulation parameters
• Select signals for recording
• Initiate code generation

LAYER 2: PUBLIC API
• Exports all block classes (VectorGain, VectorIntegrator, etc.)
• Simulation engine interface
• Utility functions for validation
• Version information

LAYER 3: CORE FUNCTIONALITY
• Block Layer: All block implementations (sources, processing, dynamic)
• Engine Layer: Simulation orchestration, loop detection, scope recording
• Generator Layer: Code generation for C/Cython targets

LAYER 4: EXECUTION TARGETS
• Python Runtime: Development and testing
• C/C++ Backend: Production deployment on MCUs
• FMU Runtime: Co-simulation with external models
"""
pdf.chapter_body(layers)

# ======================================================================
# CHAPTER 3: HOW EMBEDSIM WORKS
# ======================================================================
pdf.add_page()
pdf.chapter_title('3. HOW EMBEDSIM WORKS')

pdf.section_title('3.1 The Execution Flow')
flow_diagram = """
┌─────────────────────────────────────────────────────────────────────────┐
│                    EMBEDSIM EXECUTION FLOW                               │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  ┌─────────┐     ┌─────────┐     ┌─────────┐     ┌─────────┐           │
│  │ BLOCK   │────▶│ CONNECT │────▶│ BUILD   │────▶│ RUN     │           │
│  │ CREATION│     │ BLOCKS  │     │ GRAPH   │     │ SIMULATION│          │
│  └─────────┘     └─────────┘     └─────────┘     └────┬────┘           │
│                                                         │                │
│                                                         ▼                │
│  ┌─────────┐     ┌─────────┐     ┌─────────┐     ┌─────────┐           │
│  │ PLOT    │◀────│ ACCESS  │◀────│ RECORD  │◀────│ STEP    │           │
│  │ RESULTS │     │ DATA    │     │ SIGNALS │     │ TIME    │           │
│  └─────────┘     └─────────┘     └─────────┘     └─────────┘           │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
"""
pdf.diagram(flow_diagram)

pdf.section_title('3.2 Phase 1: Block Creation')
block_creation = """
When you create a block, this happens internally:

from embedsim import VectorGain
gain = VectorGain("my_gain", gain=2.0)

Internal initialization:
1. Memory allocated for block object
2. Properties initialized (name, dtype, backend flag)
3. Input/output lists created (empty initially)
4. Dynamic/static flag set based on block type
5. Backend dispatcher configured (routes to Python/C)
6. Block registered in the framework (optional)

The block now exists but has no connections yet.
"""
pdf.chapter_body(block_creation)

pdf.section_title('3.3 Phase 2: Connection Building')
connection = """
When you connect blocks with the >> operator:

source >> gain >> sink

Internal process:
1. gain.inputs.append(source)  # Add source to gain's input list
2. sink.inputs.append(gain)    # Add gain to sink's input list
3. Graph edge created: source → gain → sink
4. No validation yet - graph may have cycles

The >> operator returns the downstream block, enabling chaining:
source >> gain >> sink  # Works because gain >> sink returns sink
"""
pdf.chapter_body(connection)

code_example = """
# This chaining works because >> returns the right operand
block_a = VectorConstant("a", [1.0])
block_b = VectorGain("b", 2.0)
block_c = VectorEnd("c")

result = block_a >> block_b  # result is block_b
result = block_b >> block_c  # result is block_c

# So this works:
block_a >> block_b >> block_c  # Equivalent to (block_a >> block_b) >> block_c
"""
pdf.code_block(code_example)

pdf.section_title('3.4 Phase 3: Graph Traversal')
traversal = """
Before simulation, the engine analyzes the block graph to determine execution order:

order = traverse_blocks_from_sinks_with_loops([sink])

Algorithm (two-pass DFS):

Pass 1: Identify all loop breakers
- Walk graph from sinks backwards
- Mark any block inheriting from LoopBreaker
- Used later to break cycles

Pass 2: Build topological order
- Start from sinks
- Recursively traverse inputs
- Track visiting set to detect cycles
- When hitting a loop breaker, add it but don't follow its inputs
- Build post-order list (children before parents)

Time complexity: O(V + E) where V = blocks, E = connections
Space complexity: O(V)
"""
pdf.chapter_body(traversal)

example_graph = """
Example Graph:           Execution Order:
    A ──► B             1. A (source)
    │     │             2. B
    ▼     ▼             3. C
    C ◄── D             4. D

The engine ensures that when block C executes, both A and D have already
computed their outputs and are available as inputs.
"""
pdf.chapter_body(example_graph)

pdf.section_title('3.5 Phase 4: Time-Stepping Simulation')
timestepping = """
For each time step from t=0 to t=T with step dt:

┌─────────────────────────────────────────────────┐
│                 TIME STEP LOOP                   │
├─────────────────────────────────────────────────┤
│                                                  │
│  t = 0                                           │
│  while t < T:                                    │
│      # Step 1: Compute all blocks                │
│      for block in execution_order:               │
│          inputs = [inp.output for inp in block.inputs]
│          block.compute(t, dt, inputs)            │
│                                                  │
│      # Step 2: Record signals                    │
│      scope.record(t)                             │
│                                                  │
│      # Step 3: Integrate dynamics                │
│      if solver == 'euler':                       │
│          euler_step(dynamic_blocks, t, dt)       │
│      elif solver == 'rk4':                        │
│          rk4_step(dynamic_blocks, t, dt)         │
│                                                  │
│      # Step 4: Advance time                      │
│      t += dt                                      │
│  end                                             │
│                                                  │
└─────────────────────────────────────────────────┘
"""
pdf.diagram(timestepping)

pdf.section_title('3.6 The Compute Cycle Deep Dive')
compute_cycle = """
Every block inherits from VectorBlock, which implements a dispatcher:

def compute(self, t, dt, input_values):
    if self.use_c_backend:
        return self.compute_c(t, dt, input_values)  # C path
    else:
        return self.compute_py(t, dt, input_values)  # Python path

Python Path (compute_py):
1. Validate inputs (check existence, dimensions)
2. Extract signal values from input VectorSignals
3. Apply block-specific algorithm
4. Create output VectorSignal with result
5. Store in self.output and return

C Path (compute_c):
1. Pack inputs into flat numpy array
2. Call Cython wrapper (releases GIL)
3. Wrapper calls C function
4. Get results and create VectorSignal
"""
pdf.chapter_body(compute_cycle)

py_code = """
def compute_py(self, t, dt, input_values):
    # 1. Validate inputs
    if not input_values:
        raise ValueError(f"{self.name}: No input provided")
    
    # 2. Extract signal values
    u = input_values[0].value
    
    # 3. Apply algorithm
    y = self.gain * u  # Example: gain block
    
    # 4. Create output signal
    self.output = VectorSignal(y, self.name)
    
    # 5. Return for downstream blocks
    return self.output
"""
pdf.code_block(py_code)

c_code = """
def compute_c(self, t, dt, input_values):
    # 1. Pack inputs into flat array
    u = np.empty(n_inputs, dtype=np.float64)
    u[0] = input_values[0].value[0]
    
    # 2. Call Cython wrapper (GIL released)
    self._wrapper.set_inputs(u)
    self._wrapper.compute()  # Calls C function
    
    # 3. Get results
    y = self._wrapper.get_outputs()
    
    # 4. Create output signal
    self.output = VectorSignal(y, self.name)
    return self.output
"""
pdf.code_block(c_code)

pdf.section_title('3.7 Dynamic Block Integration')
dynamic = """
Dynamic blocks have internal state that evolves over time. The engine supports
multiple ODE solvers:

EULER METHOD (1st order)
    x(t+dt) = x(t) + dx/dt * dt
    Fast, O(dt) error, good for simple systems

RK4 METHOD (4th order)
    k1 = f(x, u, t)
    k2 = f(x + k1*dt/2, u, t + dt/2)
    k3 = f(x + k2*dt/2, u, t + dt/2)
    k4 = f(x + k3*dt, u, t + dt)
    x(t+dt) = x(t) + (dt/6)*(k1 + 2k2 + 2k3 + k4)
    Most accurate, O(dt⁴) error, 4x more computations

HEUN METHOD (2nd order)
    Predictor: x_pred = x + f(x,u,t)*dt
    Corrector: x(t+dt) = x + (f(x,u,t) + f(x_pred,u,t+dt))/2 * dt
    Good compromise, O(dt²) error
"""
pdf.chapter_body(dynamic)

rk4_code = """
def _integrate_dynamics_rk4(self, t):
    # Save initial states
    initial = {b: b.state.copy() for b in self.dynamic_blocks}
    
    # k1
    for b in self.dynamic_blocks:
        b.k1 = b.get_derivative(t, self._get_inputs(b))
    
    # k2 (advance half step with k1)
    for b in self.dynamic_blocks:
        b.state = initial[b] + 0.5 * self.dt * b.k1
    self._compute_all_blocks(t + 0.5 * self.dt)
    
    for b in self.dynamic_blocks:
        b.k2 = b.get_derivative(t + 0.5 * self.dt, self._get_inputs(b))
    
    # k3 (advance half step with k2)
    for b in self.dynamic_blocks:
        b.state = initial[b] + 0.5 * self.dt * b.k2
    self._compute_all_blocks(t + 0.5 * self.dt)
    
    for b in self.dynamic_blocks:
        b.k3 = b.get_derivative(t + 0.5 * self.dt, self._get_inputs(b))
    
    # k4 (advance full step with k3)
    for b in self.dynamic_blocks:
        b.state = initial[b] + self.dt * b.k3
    self._compute_all_blocks(t + self.dt)
    
    for b in self.dynamic_blocks:
        b.k4 = b.get_derivative(t + self.dt, self._get_inputs(b))
    
    # Final update
    for b in self.dynamic_blocks:
        b.state = initial[b] + (self.dt/6.0) * (
            b.k1 + 2*b.k2 + 2*b.k3 + b.k4
        )
"""
pdf.code_block(rk4_code)

pdf.section_title('3.8 Feedback Loop Handling')
feedback = """
The algebraic loop problem occurs when blocks form a cycle without delay:

    [Sum] ──► [Gain]
      ▲        │
      │        │
      └────────┘

This creates a circular dependency: Sum needs Gain's output, Gain needs Sum's output.

Solution: Loop Breakers (e.g., VectorDelay)

    [Sum] ──► [Gain]
      ▲        │
      │        │
      └──[Delay]┘

How loop breakers work:
1. Detection: Engine identifies blocks inheriting from LoopBreaker
2. Pre-computation: Before main compute pass, calls get_loop_breaking_output()
3. Graph traversal: When DFS hits a loop breaker, stops recursion
4. Runtime: Loop breaker provides previous value, not requiring current input
"""
pdf.chapter_body(feedback)

delay_code = """
class VectorDelay(VectorBlock, LoopBreaker):
    def get_loop_breaking_output(self):
        # Called BEFORE compute pass
        # Returns value from previous timestep
        return self.last_output
    
    def compute_py(self, t, dt, input_values):
        # Output = previous input
        if self.last_output:
            val = self.last_output.value.copy()
        else:
            val = np.zeros_like(input_values[0].value)
        
        # Store current input for next time
        self.last_output = VectorSignal(input_values[0].value.copy())
        
        self.output = VectorSignal(val, self.name)
        return self.output
"""
pdf.code_block(delay_code)

# ======================================================================
# CHAPTER 4: BLOCK REFERENCE
# ======================================================================
pdf.add_page()
pdf.chapter_title('4. BLOCK REFERENCE')

pdf.section_title('4.1 Source Blocks')
pdf.subsection_title('VectorConstant')
vc = """
Outputs a constant vector signal regardless of simulation time.

Parameters:
    name: str - Unique identifier for this block
    value: Union[List[float], np.ndarray] - Constant vector value
    use_c_backend: bool = False - Whether to use C backend
    dtype: Optional - Override default dtype (float32)

Example:
    const = VectorConstant("ref", [5.0, 5.0, 5.0])
    const = VectorConstant("bias", np.array([1.0, 2.0, 3.0]))
"""
pdf.chapter_body(vc)

pdf.subsection_title('VectorStep')
vs = """
Outputs a step change at specified time.

Parameters:
    name: str - Unique identifier
    step_time: float = 0.0 - Time when step occurs (seconds)
    before_value: float = 0.0 - Value before step
    after_value: float = 1.0 - Value after step
    dim: int = 3 - Dimension of output vector
    use_c_backend: bool = False
    dtype: Optional

Example:
    step = VectorStep("load", step_time=0.5, before_value=0.0, 
                      after_value=10.0, dim=2)
"""
pdf.chapter_body(vs)

pdf.subsection_title('ThreePhaseGenerator')
tpg = """
Generates balanced three-phase sinusoidal signals (120° separation).

Parameters:
    name: str - Unique identifier
    amplitude: float = 1.0 - Peak value of each phase
    freq: float = 50.0 - Frequency in Hz (50=Europe, 60=North America)
    phase: float = 0.0 - Phase shift of phase A in radians
    use_c_backend: bool = False
    dtype: Optional

Output: [phase_a, phase_b, phase_c] where:
    phase_a = amplitude * sin(ωt + φ)
    phase_b = amplitude * sin(ωt - 2π/3 + φ)
    phase_c = amplitude * sin(ωt + 2π/3 + φ)

Example:
    gen = ThreePhaseGenerator("mains", amplitude=230.0, freq=50.0)
"""
pdf.chapter_body(tpg)

pdf.subsection_title('SinusoidalGenerator')
sg = """
Generates a single sinusoidal signal.

Parameters:
    name: str - Unique identifier
    amplitude: float = 1.0 - Peak value
    freq: float = 50.0 - Frequency in Hz
    phase: float = 0.0 - Phase shift in radians
    use_c_backend: bool = False
    dtype: Optional

Example:
    sine = SinusoidalGenerator("osc", amplitude=5.0, freq=100.0)
"""
pdf.chapter_body(sg)

pdf.subsection_title('VectorRamp')
vr = """
Generates a linearly increasing (or decreasing) signal.

Parameters:
    name: str - Unique identifier
    slope: float = 1.0 - Rate of change (units per second)
    initial_value: float = 0.0 - Starting value
    start_time: float = 0.0 - Time when ramp begins
    dim: int = 3 - Output dimension
    use_c_backend: bool = False
    dtype: Optional

Formula: y = initial_value + slope * max(0, t - start_time)

Example:
    ramp = VectorRamp("accel", slope=2.0, initial_value=0.0, start_time=1.0, dim=1)
"""
pdf.chapter_body(vr)

pdf.section_title('4.2 Processing Blocks')
pdf.subsection_title('VectorGain')
vg = """
Multiplies input by scalar or matrix gain.

Parameters:
    name: str - Unique identifier
    gain: Union[float, np.ndarray] - Gain value (scalar or matrix)
    use_c_backend: bool = False
    dtype: Optional

Examples:
    gain1 = VectorGain("amp", gain=2.0)                    # Scalar gain
    gain2 = VectorGain("matrix", gain=np.array([[1,0],[0,1]]))  # Matrix gain
"""
pdf.chapter_body(vg)

pdf.subsection_title('VectorSum')
vsum = """
Sums multiple inputs with optional signs.

Parameters:
    name: str - Unique identifier
    signs: Optional[List[float]] - Signs/weights per input (None = all +1)
    use_c_backend: bool = False
    dtype: Optional

Examples:
    sum1 = VectorSum("adder")                    # All +1
    sum2 = VectorSum("subtractor", signs=[1, -1]) # First minus second
    sum3 = VectorSum("weighted", signs=[0.5, 0.3, 0.2]) # Weighted average
"""
pdf.chapter_body(vsum)

pdf.subsection_title('VectorDelay')
vd = """
One-step delay (z⁻¹). Also serves as a loop breaker for feedback.

Parameters:
    name: str - Unique identifier
    initial: Optional[List[float]] - Initial output value (None = zeros)
    use_c_backend: bool = False
    dtype: Optional

Behavior: y(t) = u(t-1) for t>0, y(0) = initial

Example:
    delay = VectorDelay("z1", initial=[0.0, 0.0, 0.0])
"""
pdf.chapter_body(vd)

pdf.subsection_title('VectorProduct')
vp = """
Element-wise multiplication of two inputs.

Parameters:
    name: str - Unique identifier
    use_c_backend: bool = False
    dtype: Optional

Formula: output[i] = input1[i] * input2[i]

Example:
    power = VectorProduct("power")
    voltage >> power
    current >> power
"""
pdf.chapter_body(vp)

pdf.subsection_title('VectorAbs')
va = """
Absolute value of each element.

Parameters:
    name: str - Unique identifier
    use_c_backend: bool = False
    dtype: Optional

Formula: output[i] = |input[i]|

Example:
    abs_block = VectorAbs("magnitude")
"""
pdf.chapter_body(va)

pdf.subsection_title('VectorSaturation')
vsat = """
Limits signal to [lower, upper] bounds.

Parameters:
    name: str - Unique identifier
    lower: float = -1.0 - Lower saturation limit
    upper: float = 1.0 - Upper saturation limit
    use_c_backend: bool = False
    dtype: Optional

Raises: ValueError if lower >= upper

Example:
    limiter = VectorSaturation("limit", lower=-10.0, upper=10.0)
"""
pdf.chapter_body(vsat)

pdf.section_title('4.3 Dynamic Blocks')
pdf.subsection_title('VectorIntegrator')
vi = """
Continuous-time integrator: ẋ = u

Parameters:
    name: str - Unique identifier
    initial_state: Optional[List[float]] - Initial state x(0)
    dim: int = 3 - Dimension (used if initial_state is None)
    use_c_backend: bool = False
    dtype: Optional

Attributes:
    state: Current integrator state
    derivative: Current rate of change

Example:
    integrator = VectorIntegrator("int", initial_state=[0.0, 0.0, 0.0])
"""
pdf.chapter_body(vi)

pdf.subsection_title('StateSpaceBlock')
ssb = """
Continuous state-space model: ẋ = Ax + Bu, y = Cx + Du

Parameters:
    name: str - Unique identifier
    A, B, C, D: np.ndarray - State-space matrices
    initial_state: Optional[List[float]] - Initial state x(0)
    use_c_backend: bool = False
    dtype: Optional

Example - DC motor:
    A = np.array([[0, 1], [0, -1]])
    B = np.array([[0], [1]])
    C = np.array([[1, 0]])
    D = np.array([[0]])
    motor = StateSpaceBlock("motor", A, B, C, D)
"""
pdf.chapter_body(ssb)

pdf.subsection_title('TransferFunctionBlock')
tfb = """
Transfer function using controllable canonical form: H(s) = num(s)/den(s)

Parameters:
    name: str - Unique identifier
    num: List[float] - Numerator coefficients (descending powers of s)
    den: List[float] - Denominator coefficients (descending powers of s)
    dim: int = 3 - Number of independent channels
    initial_state: Optional[np.ndarray] - Initial internal states
    use_c_backend: bool = False
    dtype: Optional

Example - 2nd order low-pass filter (ωₙ=1, ζ=0.707):
    lpf = TransferFunctionBlock("lpf", num=[1.0], den=[1.0, 1.414, 1.0], dim=1)
"""
pdf.chapter_body(tfb)

pdf.subsection_title('VectorEnd')
ve = """
Sink block for recording simulation outputs.

Parameters:
    name: str - Unique identifier
    use_c_backend: bool = False
    dtype: Optional

Attributes:
    history: List[np.ndarray] - Recorded values from each time step

Example:
    sink = VectorEnd("output")
    source >> sink
    print(sink.history)  # List of recorded values
"""
pdf.chapter_body(ve)

pdf.section_title('4.4 Special Blocks')
pdf.subsection_title('ScriptBlock')
sb = """
User-defined Python script with automatic C code generation.

Parameters:
    name: str - Unique identifier
    script: str - Python code (must assign to 'output')
    parameters: Optional[Dict[str, Any]]