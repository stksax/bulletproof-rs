# bulletproof-rs
referance by: https://link.springer.com/article/10.1007/s42452-019-0989-z

run test
```` cargo test ````

test.rs simulate two situation:
1. proof a number is in 0 to 2^n
2. proof a number is in given range a to b
   
# follow this step for proof
1. generate public parameter by ````generate_parameter()```` in ````generator.rs````
2. prover use the public parameter from step1 to create proof by ````range_proof()```` in ````proof.rs````
3. verifier take the proof and verify it by ````verify_range_proof()```` in ````verify.rs```` 

and it had simulate in ````test.rs````


