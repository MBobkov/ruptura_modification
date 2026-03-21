RUPTURA: Breakthrough, IAST, and Isotherm Fitting
=================================================

This software is a simulation package to compute breakthrough curves and 
IAST mixture predictions. It has been developed at Delft University of 
Technology (Delft, The Netherlands) in 2022, in active collaboration
with the University of Amsterdam (Amsterdam, The Netherlands), Eindhoven 
University of Technology (Eindhoven, The Netherlands), Pablo de Olavide 
University (Seville, Spain), and Shell Global Solutions International B.V.
(Amsterdam, The Netherlands).

In addition to the original functionality, new branches extend RUPTURA to
cyclic PSA process simulation and multilayer bed modelling.

Features
========
* Unlimited number of components
* Fast (sub-second) IAST mixture computation
* Stable breakthrough computation, including
  - step breakthrough
  - pulse breakthrough
  - Linear Driving Force (LDF) model
  - axial dispersion
  - pressure gradient
* Automatic picture/movie generation
* Isotherm models
  - Langmuir
  - Anti-Langmuir
  - BET
  - Henry
  - Freundlich
  - Sips
  - Langmuir–Freundlich
  - Redlich–Peterson
  - Toth
  - Unilan
  - O’Brien & Myers
  - Quadratic
  - Temkin
* Fitting of raw data to isotherm models
* PSA cycle simulation for Skarstrom-type processes (CycleBranch)
* Multilayer fixed-bed adsorption with multiple isotherms (2layers)

CycleBranch: Skarstrom PSA Cycle
================================

The `CycleBranch` branch contains source code for simulating cyclic
pressure swing adsorption (PSA) processes of Skarstrom type. The 
implementation introduces new process steps with additional input 
parameters controlling timing, pressures, flow directions, and gas 
compositions for each step:

* **Adsorption**  
  Feed gas is introduced at high or intermediate pressure, and the 
  more strongly adsorbed components are taken up by the adsorbent. 
  The step is controlled by feed composition, inlet pressure, 
  temperature, flow rate, and step duration. Typical outputs include 
  effluent composition histories, bed loading profiles, and local 
  mass-transfer rates.

* **Pressurization**  
  The column is brought from a lower pressure to the adsorption 
  pressure, using feed gas, product gas, or another specified stream. 
  New parameters in the input file define the pressurization strategy 
  (gas source, pressure ramp profile, duration). This step captures 
  gas redistribution and the transient loading build-up before 
  adsorption.

* **Blowdown (Depressurization)**  
  The column pressure is reduced from the adsorption pressure to a 
  lower level (often near atmospheric or intermediate pressure) to 
  desorb previously adsorbed components. The input allows specification 
  of blowdown end pressure, vent or product direction, and step time, 
  enabling simulation of different regeneration strategies and their 
  impact on working capacity and energy consumption.

* **Purge**  
  A purge gas (often a fraction of product) is passed through the bed 
  at low pressure to further remove strongly adsorbed species. Input 
  parameters define purge flowrate, composition, direction, and 
  duration. This step models deep regeneration of the adsorbent and 
  its effect on cycle performance, purity, and recovery.

These stages can be combined into full Skarstrom PSA cycles (and 
variants) by specifying a sequence of steps and their operating 
conditions in the input file, enabling detailed cycle design and 
optimization.

2layers: Multilayer Packed-Bed Modelling
========================================

The `2layers` branch contains code for modelling fixed beds with 
multiple layers of adsorbent. Instead of a single adsorbent with one 
isotherm model, the column can be represented as a stack of several 
layers, each with its own adsorption isotherm and transport properties.

Key capabilities include:

* **Multilayer packing**  
  The bed can be divided into multiple axial segments (layers), each 
  corresponding to a different adsorbent material or formulation.

* **Layer-specific isotherms**  
  Each layer can use a different isotherm model and parameters 
  (e.g., Langmuir in one layer, Sips or Toth in another), enabling 
  the simulation of hybrid or graded beds designed for improved 
  selectivity, capacity, or mass-transfer performance.

* **Coupled transport and adsorption**  
  The governing equations account for mass transfer and axial 
  dispersion across the entire bed while using the appropriate 
  isotherm and kinetic parameters in each layer. This allows 
  investigation of design strategies for multilayer beds in 
  breakthrough and cyclic processes.

The extensions in `CycleBranch` and `2layers` were implemented by  
**Matvey E. Bobkov** (Boreskov Institute of Catalysis, SB RAS;  
Novosibirsk State University).

Terms of Use
============

If you use this software for scientific publications, please cite:<br>
“RUPTURA: Simulation Code for Breakthrough, Ideal Adsorption Solution
Theory Computations, and Fitting of Isotherm Models”<br>
S. Sharma, S. Balestra, R. Baur, U. Agarwal, E. Zuidema, M. Rigutto,
S. Calero, T. J. H. Vlugt, and D. Dubbeldam,  
*Molecular Simulation* 49(9), 2023.  
[https://www.tandfonline.com/doi/full/10.1080/08927022.2023.2202757](https://www.tandfonline.com/doi/full/10.1080/08927022.2023.2202757)

If you use the Skarstrom PSA cycle and/or multilayer bed functionality
from the `CycleBranch` or `2layers` branches, please additionally 
acknowledge the contribution of **Matvey E. Bobkov**  
(Boreskov Institute of Catalysis SB RAS, Novosibirsk State University).

Authors
=======

Shrinjay Sharma,        Delft University of Technology, The Netherlands<br>
Youri Ran,              University of Amsterdam, The Netherlands<br>
Salvador R. G. Balestra, Pablo de Olavide University, Spain<br>
Richard Baur,           Shell Global Solutions International B.V., The Netherlands<br>
Umang Agarwal,          Shell Global Solutions International B.V., The Netherlands<br>
Eric Zuidema,           Shell Global Solutions International B.V., The Netherlands<br>
Marcello Rigutto,       Shell Global Solutions International B.V., The Netherlands<br>
Sofia Calero,           Eindhoven University of Technology, The Netherlands<br>
Thijs J. H. Vlugt,      Delft University of Technology, The Netherlands<br>
David Dubbeldam,        University of Amsterdam, The Netherlands<br>
Matvey E. Bobkov,       Boreskov Institute of Catalysis SB RAS / Novosibirsk State University, Russia<br>

Compilation
===========

```bash
cmake . -B build
cmake --build build
