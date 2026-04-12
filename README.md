#### Collision Testbench

Testbench for contact algorithms. Evaluates the stability and performance.

Implemented algorithms: 
- [x] xpbd ^[1]
- [x] vbd ^[5]
- [ ] avbd ^[2]
- [ ] gauss-newton implicit
- [ ] analytically projected newton ^[3]
- [x] preconditioned gradient descent (primal solver in ^[4])  

#### Prerequisites 

```
pip install warp-lang libigl==2.5.1 usd-core
```

#### Command

To run xpbd rigid body chain net simulation, use: 

```
python links.py
```

Uses primal solver (`PrimalRbd`) by default. `XPBDRbd` refers to the xpbd solver. 

To run cosserat YLS, use:

```
python -m cosserat.yarn
```


To generate ee tests: 
```
python -m psd.gen_test_ee --n 5000 --out ee
```

To test against IPC Hessian:
```
python -m psd.ee_ipc
```
#### Performance

Tested on RTX 3080. Python 3.10.14. YLS takes 5ms to simulate 1ms. 

| method                            | real time  | sim time | iterations | early termination | 
|---|---|---|---|---|
| xpbd                              |~ 1ms    | 2ms       | 2          | true |
| preconditioned gradient descent   | ~2ms    | 8ms       | 16          | true |
|PGD (w/ line search)               | ~5ms     | 8ms       | 16          | true |
|vbd                                | ~5ms     | 8ms        | 16         | true |
#### Reference

[1]: Detailed Rigid Body Simulation with Extended Position Based Dynamics
[2]: Augmented Vertex Block Descent
[3]: A Unified Analysis of Penalty-Based Collision Energies
[4]: Primal/Dual Descent Methods for Dynamics
[5]: Vertex Block Descent


#### Coding Agent Instructions 

1. Don't create individual environments to run the scripts. A conda environment named base is created for you to run this script. Don't modify the packages inside the environment. 
2. Use self defined "scalar" types in warp context so that all code supports switching between single and double precision easily.
3. Warp does not do type conversion inside warp.kernel or warp.func, even between double and single float. Use explicit type conversion like scalar(1.0) for constants. 
