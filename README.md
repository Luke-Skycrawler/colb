#### Collision Testbench

Testbench for contact algorithms. Evaluates the stability and performance (sim time / real time).

Implemented algorithms: 
[x] xpbd ^[1]
[ ] avbd ^[2]
[ ] gauss-newton implicit
[ ] analytically projected newton ^[3]

#### Command

To run xpbd rigid body chain net simulation, use: 

```
python links.py
```

#### Reference

[1]: Detailed Rigid Body Simulation with Extended Position Based Dynamics
[2]: Augmented Vertex Block Descent
[3]: A Unified Analysis of Penalty-Based Collision Energies