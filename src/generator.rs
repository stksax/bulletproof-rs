use pasta_curves::{group::{ff::{Field, PrimeField}, Group}, pallas};
use std::{ops:: Mul, u128};

#[derive(Clone)]
pub struct GenerateParameter{
    pub g_vec: Vec<pasta_curves::Ep>,
    pub h_vec: Vec<pasta_curves::Ep>,
    pub h_single_point: pasta_curves::Ep,
    pub len: usize
}

pub fn generate_parameter(g_scalar: u128, h_scalar: u128, len: usize) -> GenerateParameter{
        let generator_point = pallas::Point::generator();
        let g_vec = constract_ecc_vec(g_scalar, len);
        let h_scalar_fq = pallas::Scalar::from_u128(h_scalar);
        let h_single_point = generator_point.mul(h_scalar_fq);
        let h_vec = constract_ecc_vec(h_scalar, len);

        let out = GenerateParameter{
            g_vec,
            h_vec,
            h_single_point,
            len
        };

        out
}


pub fn to_binary_vec(num: u128, len: usize) -> Vec<pallas::Scalar> {
    let mut out = Vec::with_capacity(len);  
    let mut n = num;
    
    for _ in 0..len {
        out.push(pallas::Scalar::from_u128((n % 2) as u128));
        n /= 2;
    }

    out
}

// pub fn to_inverse_vec(num: Vec<pallas::Scalar>, len: usize) -> Vec<pallas::Scalar> {
//     let mut out = Vec::new();
//     let one = pallas::Scalar::one();
//     for i in 0..len {
//         out.push(num[i] - one);
//     }

//     out
// }

pub fn to_inverse_vec(num: &Vec<pallas::Scalar>, len: usize) -> Vec<pallas::Scalar> {
    let one = pallas::Scalar::one();
    let mut out = num.clone(); 

    for i in &mut out[..len] {
        *i = *i - one; 
    }

    out
}


pub fn constract_ecc_vec(num: u128, len: usize) -> Vec<pallas::Point> {
    let mut out = Vec::new();
    for i in 0..len{
        if i == 0{
            out.push(pallas::Point::generator());
        }else{
            out.push(out[i-1].mul(pallas::Scalar::from_u128(num)));
        }
    }
    out
}

pub fn generator_arbitrary_vec_with_fix_len (len: usize) -> Vec<pallas::Scalar> {
    let mut out = Vec::new();
    for _ in 0..len{
        out.push(pallas::Scalar::random(rand::rngs::OsRng));
    }
    out
}

