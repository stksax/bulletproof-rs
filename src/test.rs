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

    fn range_test(){
        //test for is 43 in range of 23 to 63
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

        let secret_low: u128 = 23;
        let commit_parameter_low = CommitParameter{
            secret_v: secret_low,
            secret_r: random_r,
            x: x,
            y: y,
            z: z,
            len: 7,
            generate_parameter: generate_parameter.clone()
        };

        let secret_high: u128 = 63;
        let commit_parameter_high = CommitParameter{
            secret_v: secret_high,
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
        let (commit_low, response_low) = commit_parameter_low.proof();
        let (commit_high, response_high) = commit_parameter_high.proof();


        let generator_point = pallas::Point::generator();
        let vg = generator_point.mul(pallas::Scalar::from_u128(secret));
        let rh = h_single_point.mul(pallas::Scalar::from_u128(random_r));
        let commit_origin = vg.add(rh);

        let verify_parameter = VerifyParameter{ 
            commit_origin: commit_origin,
            commit: commit,
            response: response,
            x: x, 
            g_vec: g_vec.clone(),
            z: z, 
            h_vec: h_vec.clone(),
            y: y, 
            len: 7,
            h_single_point: h_single_point
        };

        let neg_low = generator_point.mul(pallas::Scalar::from_u128(secret_low).neg());
        let low_point = commit_origin.add(neg_low);
        let verify_parameter_low = VerifyParameter{ 
            commit_origin: low_point,
            commit: commit_low,
            response: response_low,
            x: x, 
            g_vec: g_vec.clone(),
            z: z, 
            h_vec: h_vec.clone(),
            y: y, 
            len: 7,
            h_single_point: h_single_point
        };

        let two: u128 = 2;
        let exp2 = two.pow(7 as u32);
        let neg_high = generator_point.mul(pallas::Scalar::from_u128(secret_high).neg());
        let exp2_point = generator_point.mul(pallas::Scalar::from_u128(exp2));
        let high_point = commit_origin.add(neg_low).add(exp2_point);
        let verify_parameter_high = VerifyParameter{ 
            commit_origin: high_point,
            commit: commit_high,
            response: response_high,
            x: x, 
            g_vec: g_vec.clone(),
            z: z, 
            h_vec: h_vec.clone(),
            y: y, 
            len: 7,
            h_single_point: h_single_point
        };

        verify_parameter.verify();
    }
}