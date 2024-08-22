use std::{ops::{Add, Mul}, u128};
use pasta_curves::{group::{ff::{Field, PrimeField}, Group}, pallas};

use crate::verify::*;
use crate::generator::*;
use crate::proof::*;


#[cfg(test)]
mod tests{
    use super::*;
    #[test]
    fn test(){
        let random_num1: u128 = 432524;
        let random_num2: u128 = 65768;
        let random_r: u128 = 64482;

        let generate_parameter = generate_parameter(random_num1, random_num2, 7);
        
        //to-do change to hash value
        let y = pallas::Scalar::random(rand::rngs::OsRng);
        let z = pallas::Scalar::random(rand::rngs::OsRng);
        let x = pallas::Scalar::random(rand::rngs::OsRng);

        //secret < 2**7
        let secret: u128 = 43;
        let commit_parameter = CommitParameter{
            secret_v: secret,
            secret_r: random_r,
            x: x,
            y: y,
            z: z,
            len: 7,
            generate_parameter: generate_parameter.clone()
        };

        let g_vec = generate_parameter.g_vec;
        let h_vec = generate_parameter.h_vec;
        let h_single_point = generate_parameter.h_single_point;

        let (commit, response) = commit_parameter.proof();


        let generator_point = pallas::Point::generator();
        let vg = generator_point.mul(pallas::Scalar::from_u128(secret));
        let rh = h_single_point.mul(pallas::Scalar::from_u128(random_r));
        let commit_origin = vg.add(rh);

        let verify_parameter = VerifyParameter{ 
            commit_origin: commit_origin,
            commit: commit,
            response: response,
            x: x, 
            g_vec: g_vec,
            z: z, 
            h_vec: h_vec,
            y: y, 
            len: 7,
            h_single_point: h_single_point
        };

        verify_parameter.verify();
    }
}