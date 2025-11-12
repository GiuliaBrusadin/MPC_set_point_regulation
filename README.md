Torino, IT, 12th November 2025

# On solving the closed-loop setpoint regulation problem for the replicator equation via  non-linear MPC - related code
#### Giulia Brusadin, Michele Pagone, Lorenzo Zino, and Alessandro Rizzo

The code in this repository has been developed to run the simulations need for the article *On solving the closed-loop setpoint regulation problem for the replicator equation via  nonlinear MPC*, written by G. Brusadin, M. Pagone, L. Zino, and A. Rizzo, currently submitted and under review for the ECC 2026. The paper will be available for consultation after publication.

## System requirements
To run the code in this repository the following Pyhton version and libraries are needed:
* Python v3.8.10
* CasADi v3.7.2
* numpy v1.24.4
* math
* matplotlib v3.7.5

## Files overview

### function_file
This file contains the functiones developed fot he other scripts to work, therefore needs to be run first.

### simulation_without_control
This script runs the discretized system dynamics of a two players game with replicator equation as update strategy. There is no control added to the system and the obtained dynamic is then shown in a plot.

### simulation_MPC_control
This script runs the same system as above, but iwth the MPC based controlled described in the article. The uncontrolled state, controlled state and control gain (scaled) are then shown in a plot.



