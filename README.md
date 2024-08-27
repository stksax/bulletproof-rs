# bulletproof-rs
referance by: https://link.springer.com/article/10.1007/s42452-019-0989-z

run test
```` cargo test ````

test.rs simulate two situation about proof a number is in 0~2^n and number is in given range a~b
# pseudocode

### Using random numbers to generate public parameters
   public parameters = generate_parameter(random numbers)

   ### prover use public parameters to construct public data and broadcast
   public data= pedeson commit(secret , public parameters)

   ### verifier set range
   range.send_to(prover, range)
 ### prover construct commit with given range
   ### 2 ^ (len-1) < range <  2 ^ len
   (commit, response) = commit_parameter.proof(secret, commit_parameter) 
   (commit_range_lower bound, response_range_lower bound) = commit_parameter.proof(secret - lower bound, commit_parameter)
   (commit_range_upper bound, response_range_upper bound) = commit_parameter.proof(secret - upper bound + 2^len, commit_parameter)

   ### verifier verify
   assert_equal(commit - lower bound , commit_range_lower bound)
   assert_equal(commit  - upper bound + 2^len , commit_range_upper bound)
   bool1 = verify(commit, response)
   bool2 = verify(commit_range_lower bound, response_range_lower bound)
   bool3 = verify(commit_range_upper bound, response_range_upper bound)
   assert_true(bool1, bool2, bool3)



