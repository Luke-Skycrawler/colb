#### Collision Testbench

Testbench for contact algorithms. Evaluates the stability and performance.

Implemented algorithms: 
- [x] xpbd ^[1]
- [x] vbd ^[5]
- [ ] avbd ^[2]
- [ ] gauss-newton implicit
- [ ] analytically projected newton ^[3]
- [x] preconditioned gradient descent (primal solver in ^[4])  

#### Command

To run xpbd rigid body chain net simulation, use: 

```
python links.py
```

Uses primal solver (`PrimalRbd`) by default. `XPBDRbd` refers to the xpbd solver. 


#### Performance

Tested on RTX 3080.

| method                            | real time  | sim time | iterations | early termination | 
|---|---|---|---|---|
| xpbd                              |~ 1ms    | 2ms       | 2          | true |
| preconditioned gradient descent   | ~1ms    | 4ms       | 8          | true |
|PGD (w/ line search)               | ~2ms     | 4ms       | 4          | true |
|vbd                                | ~4ms     | 4ms        | 8         | true |
#### Reference

[1]: Detailed Rigid Body Simulation with Extended Position Based Dynamics
[2]: Augmented Vertex Block Descent
[3]: A Unified Analysis of Penalty-Based Collision Energies
[4]: Primal/Dual Descent Methods for Dynamics
[5]: Vertex Block Descent